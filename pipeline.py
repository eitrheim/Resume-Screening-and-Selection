from __future__ import absolute_import

import sys


def pipeline(jobID='abcd123', topX=10, root_file_path='/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/'):
    sys.path.append(root_file_path + "Resume-Parser-master-new/bin")
    sys.path.append(root_file_path + "Resume-Parser-master-new")
    sys.path.append(root_file_path + "Resume-Parser-JOBS/bin")
    sys.path.append(root_file_path + "Resume-Parser-JOBS")
    sys.path.append(root_file_path + "models/content_based")
    import main
    import OneHotRESUMES
    import mainJOBS
    import OneHotJOBS
    import final_model

    main.main(root_file_path)
    print('new resumes converted to text')
    OneHotRESUMES.onehot(root_file_path)
    print('one hot created for resumes')
    mainJOBS.main(root_file_path)
    print('job descriptions parsed')
    OneHotJOBS.onehot(root_file_path)
    print('one hot created for job descriptions')
    ranks, jd = final_model.rank(jobID, topX, root_file_path)
    print('rank candidates')

    print('Job Description:', jd)
    print(ranks)
    print('done')


if __name__ == '__main__':
    pipeline('cash123', 100, '/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/')
