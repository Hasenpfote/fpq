#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import subprocess


def main():
    self_script = os.path.basename(__file__)
    scripts = [os.path.basename(filepath) for filepath in glob.glob('./*.py')]
    scripts.remove(self_script)
    num_scripts = len(scripts)

    print('*** Run all scripts.')

    for i, script in enumerate(scripts):
        try:
            print('*** {}/{}: `{}` start.'.format(i+1, num_scripts, script))
            subprocess.check_call(['python', script])
            print('*** end.')
        except subprocess.CalledProcessError as e:
            print(e)

    print('*** Done.')

    
if __name__ == '__main__':
    main()
