import threading

from animaEasy.datasets import inspectDataset
from animaEasy.tasks import TaskQueue


def testFilesystemWorkIsDispatchedOffTheUiThread(sampleDataset):
    tasks = TaskQueue()
    reports = []
    errors = []
    callbackThreads = []
    mainThread = threading.get_ident()

    def work():
        return threading.get_ident(), inspectDataset(str(sampleDataset))

    def completed(result):
        reports.append(result)
        callbackThreads.append(threading.get_ident())

    try:
        tasks.submit(work, completed, errors.append)
        tasks.pending[0].future.result(timeout=10)
        assert not reports
        tasks.poll()
        assert not errors
        assert len(reports) == 1
        assert reports[0][0] != mainThread
        assert reports[0][1].imageCount == 3
        assert callbackThreads == [mainThread]
        assert not tasks.busy
    finally:
        tasks.close()


def testWorkerErrorsAreDeliveredWithoutBeingSwallowed(sandboxFolder):
    tasks = TaskQueue()
    results = []
    errors = []
    tasks.submit(lambda: inspectDataset(str(sandboxFolder / "missing")), results.append, errors.append)
    tasks.close()
    assert not results
    assert len(errors) == 1
    assert "not found" in str(errors[0])
