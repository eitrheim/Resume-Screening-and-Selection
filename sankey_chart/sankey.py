import pandas as pd
import plotly


def genSankey(df, cat_cols=[], value_cols='', title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#646464', '#646464', '#646464', '#646464', '#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []

    reds = ['Selected other more qualified candidate - 16102',
            'Basic Qualifications - 15556', 'Negative - 31658',
            'Not Relevant - 90997']
    grays = ['Position Cancelled - 6335', 'Hiring Restrictions - 1311',
             'Hired For Another Job - 923', 'Hiring Policy - 831',
             'Withdrew or Hired For Another Job - 1351', 'Neutral - 59339',
             'Position Cancelled - 6335', 'Hiring Restrictions/Policy - 2142',
             'Not Reviewed Not Considered - 49939']
    greens = ['Skills or Abilities - 3634',
              'Salary Expectations too high - 2224', 'Completion - 1526',
              'Phone Screen - 336', 'Schedule Interview - 94',
              'Offer Rejected - 70', 'No Show (Interview / First Day) - 14',
              'Offer - 9', 'Second Round Interview - 6',
              'Background Check - 2', 'Revise Offer - 2',
              'Final Round Interview - 1', 'Positive - 16512',
              'Voluntary Withdrew - 428',
              'Salary Expectations too high - 2224',
              'Skills or Abilities - 3634', 'Review - 8166',
              'Relevant - 16512', 'Positive - 16084',
              'Completion, Offers, Background Check - 1609',
              'Interview (various stages) - 451']

    for group in labelList:
        if group in greens:
            colorList.append('#90C79E')
        elif group in grays:
            colorList.append('#8E8D8D')
        elif group in reds:
            colorList.append('#E97070')
        else:
            colorList.append('#FF8000')

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            sourceTargetDf.columns = ['source', 'target', 'count']
        else:
            tempDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            tempDf.columns = ['source', 'target', 'count']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source', 'target']).agg({'count': 'sum'}).reset_index()

    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))

    flow_colors = []
    for i in sourceTargetDf['source']:
        if i in greens:
            flow_colors.append('#DFE9DF')
        elif i in grays:
            flow_colors.append('#E2E2E2')
        elif i in reds:
            flow_colors.append('#F5E3E3')
        else:
            flow_colors.append('#FF8000')

    # creating the sankey diagram
    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5),
            label=labelList,
            color=colorList,
            hoverinfo='skip'),
        textfont=dict(
            family='Times New Roman',
            size=14),
        arrangement='snap',
        link=dict(
            source=sourceTargetDf['sourceID'],
            target=sourceTargetDf['targetID'],
            value=sourceTargetDf['count'],
            hoverinfo='skip',
            color=flow_colors))

    layout = dict(
        title=title,
        font=dict(
            size=10))

    fig = dict(data=[data], layout=layout)
    return fig


df = pd.read_csv('flows2.csv')
df['Amount'] = df['Amount'].apply(lambda x: x.replace(',', ''))
df['Amount'] = df['Amount'].astype(int)
df['Source OG'] = df['Source']
df['Source'] = df['Source'] + " - " + df['Amount'].astype(str).values
df['Target OG'] = df['Target']
df['Target'] = df['Target'].map({'Neutral': 'Neutral - 59339',
                                 'Negative': "Negative - 31658",
                                 'Relevant': "Relevant - 16512",
                                 'Not Relevant': "Not Relevant - 90997",
                                 'Positive': 'Positive - 16512'})

fig = genSankey(df,cat_cols=['Source', 'Target'],
                value_cols='Amount', title='Latest Recruiting Stage - Is It A Relevant Recommendation?')
plotly.offline.plot(fig, validate=False)
