import pickle
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE

filename = 'glove2word2vec_model.sav'
model = pickle.load(open(filename, 'rb'))


def append_list(sim_words, words):
    list_of_words = []

    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words


input_word = 'bag'
user_input = [x.strip() for x in input_word.split(',')]
result_word = []

for words in user_input:
    sim_words = model.most_similar(words, topn=5)
    sim_words = append_list(sim_words, words)
    result_word.extend(sim_words)

similar_word = [word[0] for word in result_word]
similarity = [word[1] for word in result_word]
similar_word.extend(user_input)
labels = [word[2] for word in result_word]
label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
color_map = [label_dict[x] for x in labels]


def display_tsne_scatterplot_3D(model, user_input, words, label, color_map, perplexity=0,
                                learning_rate=0, iteration=0, topn=5, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])

    three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                     n_iter=iteration).fit_transform(word_vectors)[:, :3]

    data = []

    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            z=three_dim[count:count + topn, 2],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
        data.append(trace)
        count = count + topn
    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )
    data.append(trace_input)
    # 配置布局
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )
    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


display_tsne_scatterplot_3D(model, user_input, similar_word, labels, color_map, 5, 500, 10000)
