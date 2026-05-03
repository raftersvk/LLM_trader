import logging
import os
import sys

from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from typing import Optional

from src.config.loader import config

install_rich_traceback()


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, log_dir, log_filename_prefix, logger_name, *args, is_error_handler=False, **kwargs):
        self.log_dir = log_dir
        self.log_filename_prefix = log_filename_prefix
        self.logger_name = logger_name
        self.is_error_handler = is_error_handler
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        current_date = datetime.now().strftime("%Y_%m_%d")

        # Always use the main logger directory, even for errors
        current_log_dir = os.path.join(self.log_dir, self.logger_name, current_date)

        if self.is_error_handler:
            # Force filename to be errors.log for error handler
            current_filename = os.path.join(current_log_dir, "errors.log")
        else:
            current_filename = os.path.join(current_log_dir, f"{self.log_filename_prefix}{self.logger_name}.log")

        # Normalize paths for consistent comparison across platforms
        current_filename_norm = os.path.normpath(current_filename)
        try:
            basefilename_norm = os.path.normpath(self.baseFilename)
        except AttributeError:
            basefilename_norm = None

        if basefilename_norm != current_filename_norm:
            # Close previous stream if it exists before opening a new one
            try:
                self.stream.close()
            except AttributeError:
                pass
            except Exception:
                pass

            self.baseFilename = current_filename_norm
            if not os.path.exists(current_log_dir):
                os.makedirs(current_log_dir, exist_ok=True)
            self.stream = self._open()

        super().emit(record)


class _SymbolFilter(logging.Filter):
    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    def filter(self, record):
        if self._logger.symbol_context:
            if record.msg and isinstance(record.msg, str):
                record.msg = f"[{self._logger.symbol_context}] {record.msg}"
        return True


class Logger(logging.Logger):
    def __init__(self, logger_name: str = '', log_filename_prefix: str = '', log_dir: str = None,
                 logger_debug: bool = False) -> None:
        sanitized_name = logger_name.replace('/', '_').replace('\\', '_')

        level = logging.DEBUG if logger_debug else logging.INFO
        super().__init__(sanitized_name, level)

        self.log_filename_prefix = log_filename_prefix
        self.symbol_context: Optional[str] = None

        if log_dir is None:
            self.log_dir = config.LOG_DIR
        else:
            self.log_dir = log_dir

        self.date_format = "%d.%m.%Y %H:%M:%S"

        self._setup_logger()
        self.debug("Logger %s initialized with log directory: %s", sanitized_name, self.log_dir)

    def set_symbol(self, symbol: Optional[str]) -> None:
        self.symbol_context = symbol

    def getChild(self, suffix):
        child = Logger(self.name + '.' + suffix, self.log_filename_prefix, self.log_dir, self.level == logging.DEBUG)
        child.symbol_context = self.symbol_context
        for handler in child.handlers[:]:
            handler.filters = [f for f in handler.filters if not isinstance(f, _SymbolFilter)]
            handler.addFilter(_SymbolFilter(child))
        return child

    def _get_log_dir(self, current_date: str) -> str:
        log_dir = os.path.join(self.log_dir, self.name, current_date)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_log_filename(self, log_dir: str, suffix: str = '') -> str:
        prefix = self.log_filename_prefix if self.log_filename_prefix else ""
        name = self.name if self.name else "default"
        return os.path.join(log_dir, f"{prefix}{name}{suffix}.log")

    def _plain_formatter(self) -> logging.Formatter:
        format_string = "[{asctime}] {message}" if self.level == logging.DEBUG else "[{asctime}] {message}"
        return logging.Formatter(format_string, datefmt=self.date_format, style="{")

    def _setup_logger(self) -> None:
        current_date = datetime.now().strftime("%Y_%m_%d")
        log_dir = self._get_log_dir(current_date)
        error_log_dir = log_dir

        if not self.handlers:
            self._add_console_handler()
            self._add_file_handler(log_dir)
            self._add_error_file_handler(error_log_dir)

    def _add_console_handler(self):
        console = Console(color_system="auto", width=180)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=False,
            show_time=True,
            show_path=self.level == logging.DEBUG,
            log_time_format='[%d/%m/%y %H:%M:%S]'
        )
        rich_handler.setLevel(self.level)
        rich_handler.addFilter(_SymbolFilter(self))
        self.addHandler(rich_handler)

    def _add_file_handler(self, log_dir):
        log_filename = self._get_log_filename(log_dir)
        file_handler = DailyRotatingFileHandler(
            log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=False,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._plain_formatter())
        file_handler.addFilter(_SymbolFilter(self))
        file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        file_handler.rotator = lambda source, _dest: self._log_rotator(source)
        self.addHandler(file_handler)

    def _add_error_file_handler(self, error_log_dir):
        error_log_filename = os.path.join(error_log_dir, "errors.log")
        error_file_handler = DailyRotatingFileHandler(
            error_log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=True,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(self._plain_formatter())
        error_file_handler.addFilter(_SymbolFilter(self))
        error_file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        error_file_handler.rotator = lambda source, _dest: self._log_rotator(source)
        self.addHandler(error_file_handler)

    def _log_rotator(self, source):
        new_date = datetime.now().strftime("%Y_%m_%d")
        # _get_log_dir no longer accepts is_error, it returns the main directory
        new_dir = self._get_log_dir(new_date)
        new_file = os.path.join(new_dir, os.path.basename(source))
        open(new_file, 'a', encoding='utf-8').close()

    def close(self) -> None:
        """Close all handlers and release resources."""
        for handler in self.handlers[:]:
            try:
                handler.close()
                self.removeHandler(handler)
            except Exception:
                pass

    def install_crash_handler(self) -> None:
        """Install sys.excepthook and threading.excepthook to route crashes to errors.log."""
        logger_ref = self

        def _handle_exception(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            logger_ref.critical(
                "Unhandled exception (process crash)",
                exc_info=(exc_type, exc_value, exc_tb),
            )

        sys.excepthook = _handle_exception

        import threading
        def _handle_thread_exception(args):
            if args.exc_type is SystemExit:
                return
            logger_ref.critical(
                "Unhandled exception in thread '%s'",
                args.thread.name if args.thread else "unknown",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        threading.excepthook = _handle_thread_exception
