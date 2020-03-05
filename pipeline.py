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
    ranks, jd, all_features = final_model.rank(job_id, top_x, root_file_path)
    print('Candidates ranked.\n')

    print('Job Description:', jd)
    print(ranks)

    import matplotlib.pyplot as plt


    def plot_mds(mean_vec, job_id, ranks):
        from sklearn.manifold import MDS

        jd_row = mean_vec[mean_vec.ID == job_id].index

        mds = MDS(n_components=2, random_state=1)
        pos = mds.fit_transform(mean_vec.drop(['ID', 'ReqID'], axis=1))
        xs, ys = pos[:, 0], pos[:, 1]
        for x, y in zip(xs, ys):
            plt.scatter(x, y)
        #    plt.text(x, y, name)
        plt.scatter(xs[jd_row], ys[jd_row], c='Red', marker='+')
        plt.text(xs[jd_row], ys[jd_row], 'JD')
        for i in mean_vec.index:
            if i == jd_row:
                pass
            elif mean_vec.ID[i] in ranks['Candidate ID'].values:
                plt.text(xs[i], ys[i], mean_vec.ID[i], fontsize=6)
            else:
                pass
        plt.suptitle('MDS')
        plt.grid()
        plt.savefig('distance_MDS_improved.png')
        plt.show()

    def plot_pca(mean_vec, job_id, ranks):
        from sklearn.decomposition import PCA

        jd_row = mean_vec[mean_vec.ID == job_id].index

        pca = PCA(n_components=2)
        X = pos = pca.fit_transform(mean_vec.drop(['ID', 'ReqID'], axis=1))
        xs, ys = X[:, 0], X[:, 1]
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(xs[jd_row], ys[jd_row], c='Red', marker='+')
        plt.text(xs[jd_row], ys[jd_row], 'JD')
        for i in mean_vec.index:
            if i == jd_row:
                pass
            elif mean_vec.ID[i] in ranks['Candidate ID'].values:
                plt.text(xs[i], ys[i], mean_vec.ID[i], fontsize=6)
            else:
                pass
        plt.grid()
        plt.suptitle('PCA')
        plt.savefig('distance_PCA_improved.png')
        plt.show()

    plot_mds(all_features, 'abcd123', ranks.head(10))
    plot_pca(all_features, 'abcd123', ranks.head(10))

    print('done')


if __name__ == '__main__':
    pipeline('abcd123', 10, '/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/')
