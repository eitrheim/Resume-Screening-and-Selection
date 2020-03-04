from __future__ import absolute_import
import sys


def pipeline(job_id: str, top_x: int, root_file_path: str):
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
    print('New resumes converted to text.')
    OneHotRESUMES.onehot(root_file_path)
    print('One hot created for resumes.')
    mainJOBS.main(root_file_path)
    print('Job descriptions parsed.')
    OneHotJOBS.onehot(root_file_path)
    print('One hot created for job descriptions.')
    ranks, jd = final_model.rank(job_id, top_x, root_file_path)
    print('Candidates ranked.\n')

    print('Job Description:', jd)
    print(ranks)


if __name__ == '__main__':
    pipeline('abcd123', 100, '/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/')
