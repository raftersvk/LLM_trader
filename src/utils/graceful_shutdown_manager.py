import asyncio
import signal
import sys

PYQT_AVAILABLE = False

class GracefulShutdownManager:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        logger=None,
    ):
        self.loop = loop
        self.logger = logger
        self._callbacks = []
        self._shutting_down = False

    def setup_signal_handlers(self):
        # Only set up signal handlers on Unix systems
        # On Windows, let KeyboardInterrupt propagate naturally
        if sys.platform != 'win32':
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.loop.add_signal_handler(sig, lambda s=sig, *args: self.handle_signal(s))

    def register_shutdown_callback(self, callback):
        """Register a callback to be executed during graceful shutdown."""
        if asyncio.iscoroutinefunction(callback):
            self._callbacks.append(callback)
        else:
            self._callbacks.append(callback)

    def handle_signal(self, sig: int):
        if self._shutting_down:
            if self.logger:
                self.logger.debug("Signal %s ignored - already shutting down", sig)
            return

        if self.logger:
            self.logger.info("Signal %s received - initiating shutdown...", sig)
        else:
            print(f"Received signal {sig}, initiating shutdown...")

        if self.loop.is_running() and not self.loop.is_closed():
            self.loop.create_task(self.shutdown_gracefully())

    async def shutdown_gracefully(self):
        """Execute all registered shutdown callbacks and cancel pending tasks."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self.logger:
            self.logger.info("Performing graceful shutdown...")
        else:
            print("Performing graceful shutdown...")

        try:
            # Execute registered callbacks first
            if self._callbacks:
                if self.logger:
                    self.logger.info("Executing %s shutdown callbacks...", len(self._callbacks))
                else:
                    print(f"Executing {len(self._callbacks)} shutdown callbacks...")

                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                    except Exception as e:
                        error_msg = f"Error in shutdown callback {callback}: {e}"
                        if self.logger:
                            self.logger.error(error_msg)
                        else:
                            print(error_msg)

            pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
            if pending_tasks:
                msg = f"Cancelling {len(pending_tasks)} tasks..."
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)

                for task in pending_tasks:
                    task.cancel()
                try:
                    done, still_pending = await asyncio.wait_for(asyncio.wait(pending_tasks), timeout=10.0)
                except asyncio.TimeoutError:
                    # Force cancel any remaining tasks
                    remaining = [t for t in pending_tasks if not t.done()]
                    if remaining:
                        task_details = []
                        for t in remaining:
                            try:
                                name = t.get_name()
                            except AttributeError:
                                name = str(t)
                            task_details.append(name)
                            t.cancel()
                        
                        timeout_msg = f"Force cancelling {len(remaining)} tasks: {task_details}"
                        if self.logger:
                            self.logger.warning(timeout_msg)
                        else:
                            print(timeout_msg)
                        
                        # Wait for force-cancelled tasks
                        try:
                            await asyncio.wait(remaining, timeout=2.0)
                        except Exception:
                            pass
            try:
                await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
            except (asyncio.TimeoutError, Exception) as e:
                err_msg = f"Error shutting down async generators: {e}"
                if self.logger:
                    self.logger.error(err_msg)
                else:
                    print(err_msg)
        except asyncio.CancelledError:
            # Shutdown was cancelled - finish up as best we can
            if self.logger:
                self.logger.warning("Graceful shutdown was cancelled, forcing cleanup")
            # Still try to shut down async generators
            try:
                await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
            except Exception:
                pass
            raise

