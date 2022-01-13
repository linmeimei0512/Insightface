import itertools
import threading
import time
import sys

'''
########################################
Loading Animation
########################################
'''
class LoadingAnimation:
    def __init__(self, title):
        self.title = title
        self.done = False

    def start(self):
        '''
        Start animate
        '''
        self.done = False
        t = threading.Thread(target=self._animate)
        t.start()

    def end(self):
        '''
        End animate
        '''
        self.done = True
        time.sleep(1)

    def _animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            sys.stdout.write('\r' + self.title + ' ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     ')