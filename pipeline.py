from __future__ import absolute_import

import sys


def pipeline(pdf_to_text=0, jobID='abcd123', topX=100):
    if pdf_to_text == 1:
        sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/bin")
        sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new")
        import main
        import OneHotRESUMES

        main.main()
        OneHotRESUMES.onehot()

        sys.path = list(set(sys.path))
        sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new")
        sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/bin")

    for i in sys.path:
        print(i)

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")
    import mainJOBS
    mainJOBS.main()
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")
    import OneHotJOBS
    OneHotJOBS.onehot()
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")

    sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")
    import final_model
    ranks, jd = final_model.rank(jobID, topX)
    sys.path.remove("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")

    print(jd)
    print(ranks)


if __name__ == '__main__':
    pipeline(1, 'abcd123', 10)
