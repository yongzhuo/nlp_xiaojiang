"""
Code from : https://gist.github.com/everilae/9697228
QHduan added __next__：https://github.com/qhduan/just_another_seq2seq
"""

# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
from threading import Thread
from queue import Queue

# ... or use multiprocessing versions
# WARNING: use sentinel based on value, not identity
# from multiprocessing import Process, Queue as MpQueue


class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 daemon=False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self):
        self._started = False
        try:
            while True:
                self._queue.get(timeout=0)
        except KeyboardInterrupt as e:
            raise e
        except: # pylint: disable=bare-except
            pass
        # self._thread.join()

    def __iter__(self):
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration()
        return value


def test():
    """测试"""

    def gene():
        i = 0
        while True:
            yield i
            i += 1
    t = gene()
    tt = ThreadedGenerator(t)
    for _ in range(10):
        print(next(tt))
    tt.close()
    # for i in range(10):
    #     print(next(tt))

    # for t in ThreadedGenerator(range(10)):
    #     print(t)
    # print('-' * 10)
    #
    # t = ThreadedGenerator(range(10))
    # # def gene():
    # #     for t in range(10):
    # #         yield t
    # # t = gene()
    # for _ in range(10):
    #     print(next(t))
    # print('-' * 10)



if __name__ == '__main__':
    test()
