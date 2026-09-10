import matplotlib.pyplot as plt


def plot_class_distribution(class_distribution):

    class_distribution.plot(x="Class", y="Count", kind="pie", legend=False, color=["red", "purple"])
    plt.ylabel("number of obervations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()

    class_distribution.plot(x="Class", y="Count", kind="bar", legend=False, color=["red", "purple"])
    plt.ylabel("number of obervations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()


    class_distribution.plot(x="Class", y="Count", kind="hist", legend=False, color=["red", "purple"])
    plt.ylabel("number of obervations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()
