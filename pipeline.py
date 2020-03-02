from __future__ import absolute_import

import sys


def pipeline(pdf_to_text=0, jobID='abcd123', topX=100):
    if pdf_to_text == 1:
        sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/bin")
        import main
        main.main()
        sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/bin")

        sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new")
        import OneHotRESUMES
        OneHotRESUMES.onehot()  # one hot resumes
        sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new")

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")
    import mainJOBS
    mainJOBS.main()  # parse job descriptions
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")
    import OneHotJOBS
    OneHotJOBS.onehot()  # one hot job descriptions
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")
    import final_model
    ranks, jd = final_model.rank(jobID, topX)
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")

    print(jd)
    print(ranks)


if __name__ == '__main__':
    pipeline()
