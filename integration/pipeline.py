import queue
import threading

from integration.when_which import (
    WhenWhichAdapter,
    print_decision,
)
from integration.which_executor import WhichExecutor


class RoutingPipeline:
    def __init__(self):
        self.adapter = WhenWhichAdapter()
        self.executor = WhichExecutor(
            speak=True
        )

        self.q = queue.Queue(maxsize=4)

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self.worker.start()

    def submit(self, event, frame_rgb):
        try:
            self.q.put_nowait(
                (
                    event,
                    None if frame_rgb is None else frame_rgb.copy(),
                )
            )
        except queue.Full:
            print(
                "\n⚠ Routing queue full; "
                "dropping this trigger."
            )

    def _worker(self):
        while True:
            event, frame_rgb = self.q.get()

            try:
                decision = self.adapter.route(event)

                if decision is None:
                    continue

                print_decision(
                    event,
                    decision,
                )

                self.executor.execute(
                    decision,
                    event,
                    frame_rgb,
                )

            except Exception as exc:
                print()
                print("✗ WHICH/executor error:", repr(exc))
                print()

            finally:
                self.q.task_done()
