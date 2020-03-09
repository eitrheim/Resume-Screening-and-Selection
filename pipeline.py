from __future__ import absolute_import
import sys


def pipeline(job_id: str, top_x: int, root_file_path: str, all_resumes: bool):
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

    main.main(root_file_path, job_id)
    print('New resumes converted to text.')
    OneHotRESUMES.onehot(root_file_path)
    print('One hot created for resumes.')
    mainJOBS.main(root_file_path)
    print('Job descriptions parsed.')
    OneHotJOBS.onehot(root_file_path)
    print('One hot created for job descriptions.')
    ranks, jd, all_features = final_model.rank(job_id, top_x, root_file_path, all_resumes)
    print('Candidates ranked.\n')

    print('Job Description:', jd)
    print(ranks)

    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
    import plotly.express as px


    def plot_mds(mean_vec, job_id, ranks):

        jd_row = mean_vec[mean_vec.ID == job_id].index

        mds = MDS(n_components=2, random_state=1)
        pos = mds.fit_transform(mean_vec.drop(['ID', 'ReqID'], axis=1))
        xs, ys = pos[:, 0], pos[:, 1]
        for x, y in zip(xs, ys):
            plt.scatter(x, y)
        #    plt.text(x, y, name)
        plt.scatter(xs[jd_row], ys[jd_row], c='Red', marker='+')
        plt.text(xs[jd_row], ys[jd_row], 'JD')
        for i, txt in enumerate(mean_vec.ID):
            if i == jd_row:
                pass
            else:
                plt.text(xs[i], ys[i], txt, fontsize=6)
        plt.suptitle('MDS')
        plt.grid()
        plt.savefig('distance_MDS_improved.png')
        plt.show()

    def plot_pca(mean_vec, job_id, ranks):

        jd_row = mean_vec[mean_vec.ID == job_id].index

        pca = PCA(n_components=2)
        X = pos = pca.fit_transform(mean_vec.drop(['ID', 'ReqID'], axis=1))
        xs, ys = X[:, 0], X[:, 1]
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(xs[jd_row], ys[jd_row], c='Red', marker='+')
        plt.text(xs[jd_row], ys[jd_row], 'JD')
        for i, txt in enumerate(mean_vec.ID):
            if i == jd_row:
                pass
            else:
                plt.text(xs[i], ys[i], txt, fontsize=6)
        plt.grid()
        plt.suptitle('PCA')
        plt.savefig('distance_PCA_improved.png')
        plt.show()

    def plot_3d(mean_vec, job_id, ranks):
        jd_row = mean_vec[mean_vec.ID == job_id].index

        jd_top_other = []
        for i in mean_vec.ID:
            if i == job_id:
                jd_top_other.append('JD')
            elif i in ranks['Candidate ID'].values:
                jd_top_other.append('Top Candidates')
            else:
                jd_top_other.append('Other Candidates')

        mds = MDS(n_components=3, random_state=1)
        X = mds.fit_transform(mean_vec.drop(['ID', 'ReqID'], axis=1))
        xs, ys, zs = X[:, 0], X[:, 1], X[:, 2]

        fig = px.scatter_3d(x=xs, y=ys, z=zs,
                            color=jd_top_other,
                            opacity=.7,
                            # hoverinfo='text',
                            # text=mean_vec.ID,
                            hover_name=mean_vec.ID)
        fig.show()

    # plot_mds(all_features, job_id, ranks.head(10))
    # plot_pca(all_features, job_id, ranks.head(10))
    plot_3d(all_features, job_id, ranks.head(10))

    print('done')


if __name__ == '__main__':
    pipeline('abcd123', 1000, '/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/', True)

# job id options:
# abcd123 ibm data science internship
# cash123 cashier
# wmp1234 west monroe graduate tech
# jpm1234 jpmorgan summer financial analyst
# pgi5678 fixed income analyst
# eqty373 equity analyst
# acrm789 associate account relationship mgr
