# from __future__ import absolute_import

import sys
sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")
import mainJOBS
sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/bin")
import main
sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")
import OneHotJOBS
sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new")
import OneHotRESUMES
sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")
import final_model


def pipeline():
    main.main()  # parse resumes
    OneHotRESUMES.onehot()  # one hot resumes
    mainJOBS.main()  # parse job descriptions
    OneHotJOBS.onehot()  # one hot job descriptions
    final_model('abcd123')
    print('it worked')


if __name__ == '__main__':
    pipeline()
