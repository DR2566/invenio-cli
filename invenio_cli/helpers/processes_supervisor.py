"""
ProcessesSupervisor

Features:
1) one child dies (or SIGINT/SIGTERM arrives) => all children are being terminated
2) formatted STDOUT and STDERR from children

NOTE: children must be started with
    Popen(..., stdout=PIPE, stderr=PIPE, start_new_session=True)
so each one gets its own process group (we kill whole groups).
"""
from __future__ import annotations

import os
import select
import signal
import time
from subprocess import Popen, TimeoutExpired
from typing import Callable, Final

from rich import box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

MAX_READ_SIZE: Final[int] = 2 ** 16


# --- helpers ---------------------------------------------------------------
class FormattedPrinter:
    def __init__(self, echo: Callable[[str], None], name_column_width: int) -> None:
        self._console = Console()
        self._echo = echo
        self.name_column_width = name_column_width

    def _render(self, renderable: RenderableType, **kwargs) -> str:
        """Render a rich object to a string (ANSI codes if the terminal supports it)."""
        with self._console.capture() as capture:
            self._console.print(renderable, **kwargs)
        return capture.get().rstrip("\n")

    def print_line(self, name: str, line: bytes, is_err: bool) -> None:
        text = line.decode(errors="replace").rstrip("\r")  # CRLF -> LF
        row = Text(f"{name.ljust(self.name_column_width)} │ ")
        row.append(text, style="red" if is_err else None)
        self._echo(self._render(row, soft_wrap=True))

    def print_banner(self, processes: list[SupervisedProcess]) -> None:
        table = Table(title="ProcessesSupervisor", title_justify="left", box=box.SQUARE)
        table.add_column("Process")
        table.add_column("PID")
        for process in processes:
            table.add_row(process.name, str(process.pid))
        for line in self._render(table).splitlines():
            self._echo(line)


class Pipe:
    """Self-pipe used to wake the poll loop from a signal handler."""

    def __init__(self) -> None:
        # init the file descriptors
        self.read_fd: int = -1
        self.write_fd: int = -1

        # open the pipe
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)

    def close(self) -> None:
        if self.write_fd != -1:
            os.close(self.write_fd)
            self.write_fd = -1

        if self.read_fd != -1:
            self.drain()  # possibly read remaining bytes
            os.close(self.read_fd)
            self.read_fd = -1

    def write(self, data: bytes) -> None:
        if self.write_fd == -1:
            return

        try:
            os.write(self.write_fd, data)

        except BlockingIOError:
            pass  # writing to the full pipe

    def read(self, size: int) -> bytes:
        if self.read_fd == -1:
            return b""

        try:
            return os.read(self.read_fd, size)

        except BlockingIOError:
            return b""  # reading from the empty pipe

    def drain(self) -> None:
        self.read(MAX_READ_SIZE)


class LineBuffer:
    """Accumulates raw bytes and yields only complete lines."""

    def __init__(self) -> None:
        self._rest: bytes = b""
        self._max_line_size: int = 100

    def feed(self, chunk: bytes) -> list[bytes]:
        # form lines by \n separator byte
        *lines, self._rest = (self._rest + chunk).split(b"\n")

        # do not exceed maximum _rest length
        while len(self._rest) > self._max_line_size:
            lines.append(self._rest[:self._max_line_size])
            self._rest = self._rest[self._max_line_size:]

        return lines

    def flush(self) -> list[bytes]:
        """Return the unterminated tail (on EOF)."""
        rest, self._rest = self._rest, b""
        return [rest] if rest else []


class Stream:
    """One output pipe (stdout or stderr) of a child. Autonomous: prints its own lines."""

    def __init__(self, name: str, fd: int, is_err: bool, printer: FormattedPrinter) -> None:
        self.name = name
        self.fd = fd
        self.is_err = is_err
        self._buffer = LineBuffer()
        self._printer = printer

    def _print(self, lines: list[bytes]) -> None:
        for line in lines:
            self._printer.print_line(self.name, line, self.is_err)

    def feed(self, chunk: bytes) -> None:
        self._print(self._buffer.feed(chunk))

    def flush(self) -> None:
        """Print the unterminated tail (on EOF)."""
        self._print(self._buffer.flush())


class SupervisedProcess:

    def __init__(self, process: Popen, name: str) -> None:
        self._check_input_conditions(process, name)

        self.process = process
        self.name = name
        self.pid: int = process.pid
        self.stdout_fd: int = process.stdout.fileno()
        self.stderr_fd: int = process.stderr.fileno()

    def _check_input_conditions(self, process: Popen, name: str) -> None:
        try:
            if process.stdout is None or process.stderr is None:
                raise ValueError(f"{name}: Popen must use stdout=PIPE and stderr=PIPE")
            if os.getpgid(process.pid) == os.getpgrp():
                raise ValueError(f"{name}: Popen must use start_new_session=True")

        except ProcessLookupError:
            raise ValueError(f"{name}: process is no longer running")

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def kill(self, force: bool = False) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(self.pid, sig)  # self.pid == process group id
        except (ProcessLookupError, PermissionError):
            pass  # group empty, or (macOS) only an unreaped zombie left

    def reap(self, timeout: float = 2.0) -> bool:
        """Collect the exit status & release sources. Returns False if the process is still alive."""
        try:
            self.process.wait(timeout=timeout)
            return True
        except TimeoutExpired:
            return False


# ---------------------------------------------------------------------------
class ProcessesSupervisor:
    GRACE_PERIOD: Final[float] = 10.0

    _SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGCHLD)

    def __init__(self, processes: list[tuple[str, Popen]], echo: Callable[[str], None]):
        if not processes:
            raise ValueError("no processes to supervise")

        self.echo = echo
        self.printer = FormattedPrinter(echo, max(len(name) for name, _ in processes))
        self.activation_pipe = Pipe()
        self.processes: list[SupervisedProcess] = []
        self._streams: dict[int, Stream] = {}
        self.terminating: bool = False

        # init poller
        self._poller = select.poll()

        # register activation pipe
        self._register_activation_pipe()

        # register supervised processes
        for name, process in processes:
            self._register_process(SupervisedProcess(process, name))

    # --- setup -------------------------------------------------------------
    def _register_process(self, process: SupervisedProcess) -> None:
        # register processes' streams
        for fd, is_err in ((process.stdout_fd, False), (process.stderr_fd, True)):
            self._poller.register(fd, select.POLLIN | select.POLLHUP)
            self._streams[fd] = Stream(process.name, fd, is_err, self.printer)
        self.processes.append(process)

    def _register_activation_pipe(self) -> None:
        self._poller.register(self.activation_pipe.read_fd, select.POLLIN)

    def _set_signal_handlers(self) -> None:
        for sig in self._SIGNALS:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, sig, _) -> None:
        if not self.terminating:
            self.activation_pipe.write(b"1")  # wake the poll loop
            self.terminating = True

    # --- output handling ---------------------------------------------------
    def _read_fd(self, fd: int) -> None:
        stream = self._streams[fd]
        chunk = os.read(fd, MAX_READ_SIZE)  # won't block, poll said it's readable
        if not chunk:  # EOF: the child closed its end
            stream.flush()
            self._poller.unregister(fd)
            del self._streams[fd]
            return
        stream.feed(chunk)

    def _poll(self, timeout_ms: int | None) -> None:
        """Wait up to timeout_ms for events; forward child output, consume the activation pipe if it fired."""
        for fd, _event in self._poller.poll(timeout_ms):
            if fd == self.activation_pipe.read_fd:
                self._poller.unregister(self.activation_pipe.read_fd)
            else:
                self._read_fd(fd)

    def _drain_remaining(self) -> None:
        # clean the kernel buffers after dead children's writes
        while self._streams:
            self._poll(100)

    def _wait_for_activation_signal(self) -> None:
        # forward STDOUT & STDERR until the activation signal fires
        while not self.terminating:
            self._poll(None)

    # --- termination -------------------------------------------------------
    def _kill_all_children(self, forcefully: bool = False) -> None:
        for process in self.processes:
            process.kill(force=forcefully)

    def _reap_all_zombies(self) -> None:
        for process in self.processes:
            if not process.reap():
                self.echo(f"  ✗ process {process.name} did not exit")

    def _wait_for_termination(self) -> None:
        deadline = time.monotonic() + self.GRACE_PERIOD
        while time.monotonic() < deadline:
            if not any(p.is_alive() for p in self.processes):
                break
            self._poll(100)  # keep forwarding output while they shut down

    def _terminate_processes(self) -> None:
        self.terminating = True
        self.echo("\n⚠ ProcessesSupervisor: terminating all subprocesses...")

        self._kill_all_children()  # gracefully kill children
        self._wait_for_termination()  # wait for children termination
        self._kill_all_children(forcefully=True)  # kill all still alive children forcefully
        self._reap_all_zombies()  # clean subprocesses resources
        self._drain_remaining()  # clean kernel pipes

        self.echo("✔ ProcessesSupervisor: all subprocesses terminated")

    # --- public ------------------------------------------------------------
    def supervise(self) -> None:
        try:
            self._set_signal_handlers()
            self.printer.print_banner(self.processes)
            self._wait_for_activation_signal()
        finally:
            try:
                self._terminate_processes()
            finally:
                self.activation_pipe.close()
