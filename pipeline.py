from __future__ import absolute_import

import sys

root_file_path = '/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/'


def pipeline(pdf_to_text=0, jobID='abcd123', topX=100, root_file_path=root_file_path):
    if pdf_to_text == 1:
        sys.path.append(root_file_path + "Resume-Parser-master-new/bin")
        sys.path.append(root_file_path + "Resume-Parser-master-new")
        import main
        import OneHotRESUMES

        main.main(root_file_path=root_file_path)
        # OneHotRESUMES.onehot(root_file_path=root_file_path)

        sys.path = list(set(sys.path))
        sys.path.remove(root_file_path + "Resume-Parser-master-new")
        sys.path.remove(root_file_path + "Resume-Parser-master-new/bin")

    # sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/bin")
    # import mainJOBS
    # mainJOBS.main()
    #
    # sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS")
    # import OneHotJOBS
    # OneHotJOBS.onehot()

    # sys.path.append("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/models/content_based")
    # import final_model
    # ranks, jd = final_model.rank(jobID, topX)
    #
    # print(jd)
    # print(ranks)
    print('done')


if __name__ == '__main__':
    pipeline(1, 'cash123', 100)
