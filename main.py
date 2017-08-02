import os


def main():
    # 1. use idmrg code to produce ES data with different chi
    n_theta = 120
    chi = 20
    datafile = 'data_{}_{}'.format(n_theta, chi)
    logfile = 'log_{}_{}'.format(n_theta, chi)
    os.system('python3 ./idmrg/simply.py --n_theta={} --chi={} >> {}'.
              format(n_theta, chi, datafile))

    # 2. feed the data into tf code to get series of accuracy
    os.system('python3 tf.py --data_file={}'.format(logfile))

    # 3. draw different plots according to different chi
    # one chi first:
    os.system('python3 draw.py')


    # todo:the correctness of the ES I got from idmrg code
    # todo2:use self-learning scheme to find the best point..(this way we have no more W-shape?)

if __name__ == '__main__':
    main()
