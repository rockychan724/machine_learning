# -*- coding: utf-8 -*-
# content: multithreading

import threading, time


class MyThread(threading.Thread):
    def __init__(self, lock, threadName):
        self.lock = lock
        super(MyThread, self).__init__(name=threadName)

    def run(self):
        global count
        # self.lock.acquire()
        for i in range(20):
            count += 1
            time.sleep(0.3)
            print(self.getName(), count)
        # self.lock.release()


if __name__ == '__main__':
    count = 0
    lock = threading.Lock()
    for i in range(2):
        MyThread(lock, "MyThead-" + str(i)).start()
    exit(0)
