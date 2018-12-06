import json
from tests import memTests


def main():

    config = json.load(open('config.json'))['tests']

    if config['memoryTest']:
        memTests.testMemoryBuffers()

    return

if __name__ == '__main__':
    main()