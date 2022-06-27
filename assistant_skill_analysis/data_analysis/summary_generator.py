from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from ..utils.skills_util import LABEL_FONT


def generate_summary_statistics(data, entities_list=None):
    """
    Take the workspace dictionary and display summary statistics regarding the workspace
    :param data:
    :param entities_list:
    :return:
    """

    total_examples = len(data["utterance"])
    label_frequency = Counter(data["intent"]).most_common()
    number_of_labels = len(label_frequency)
    average_example_per_intent = np.average(list(dict(label_frequency).values()))
    standard_deviation_of_intent = np.std(list(dict(label_frequency).values()))

    characteristics = list()
    characteristics.append(["Total User Examples", total_examples])
    characteristics.append(["Unique Intents", number_of_labels])
    characteristics.append(
        ["Average User Examples per Intent", int(np.around(average_example_per_intent))]
    )
    characteristics.append(
        [
            "Standard Deviation from Average",
            int(np.around(standard_deviation_of_intent)),
        ]
    )
    if entities_list:
        characteristics.append(["Total Number of Entities", len(entities_list)])
    else:
        characteristics.append(["Total Number of Entities", 0])

    df = pd.DataFrame(data=characteristics, columns=["Data Characteristic", "Value"])
    df.index = np.arange(1, len(df) + 1)
    display(Markdown("### Summary Statistics"))
    display(df)


def show_user_examples_per_intent(data):
    """
    Take the workspace dictionary and display summary statistics regarding the workspace
    :param data:
    :return:
    """

    label_frequency = Counter(data["intent"]).most_common()
    frequencies = list(reversed(label_frequency))
    df = pd.DataFrame(data=frequencies, columns=["Intent", "Number of User Examples"])
    df.index = np.arange(1, len(df) + 1)
    display(Markdown("### Sorted Distribution of User Examples per Intent"))
    display(df)


def scatter_plot_intent_dist(workspace_pd):
    """
    takes the workspace_pd and generate a scatter distribution of the intents
    :param workspace_pd:
    :return:
    """

    label_frequency = Counter(workspace_pd["intent"]).most_common()
    frequencies = list(reversed(label_frequency))
    counter_list = list(range(1, len(frequencies) + 1))
    df = pd.DataFrame(data=frequencies, columns=["Intent", "Number of User Examples"])
    df["Intent"] = counter_list

    sns.set(rc={"figure.figsize": (15, 10)})
    display(
        Markdown(
            '## <p style="text-align: center;">Sorted Distribution of User Examples \
                     per Intent</p>'
        )
    )

    plt.ylabel("Number of User Examples", fontdict=LABEL_FONT)
    plt.xlabel("Intent", fontdict=LABEL_FONT)
    ax = sns.scatterplot(x="Intent", y="Number of User Examples", data=df, s=100)


def class_imbalance_analysis(workspace_pd):
    """
    performance class imbalance analysis on the training workspace
    :param workspace_pd:
    :return:
    """

    label_frequency = Counter(workspace_pd["intent"]).most_common()
    frequencies = list(reversed(label_frequency))
    min_class, min_class_len = frequencies[0]
    max_class, max_class_len = frequencies[-1]

    if max_class_len >= 2 * min_class_len:
        display(
            Markdown(
                "### <font style='color:rgb(165, 34, 34);'> Class Imbalance Detected \
        </font>"
            )
        )
        display(
            Markdown(
                "- Data could be potentially biased towards intents with more user \
        examples"
            )
        )
        display(
            Markdown(
                "- E.g. Intent < {} > has < {} > user examples while intent < {} > has \
        just < {} > user examples ".format(
                    max_class, max_class_len, min_class, min_class_len
                )
            )
        )
        flag = True
    else:
        display(
            Markdown(
                "### <font style='color:rgb(13, 153, 34);'> No Significant Class \
        Imbalance Detected </font>"
            )
        )
        display(
            Markdown(
                "- Lower chances of inherent bias in classification towards intents with \
        more user examples"
            )
        )
        flag = False

    return flag
