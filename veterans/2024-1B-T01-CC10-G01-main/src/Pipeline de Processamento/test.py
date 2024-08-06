import unittest

def main():
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir=start_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()
