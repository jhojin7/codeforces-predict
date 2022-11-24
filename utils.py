import pandas as pd
import model as m

def accuracy_by_tag(y_test, y_pred):
    df = pd.DataFrame(index=y_test.columns,columns=["accuracy"])
    # cols = "binarysearch bruteforce datastructues dfsandsimilar dp games math strings trees implementation greedy graphs".split()
    for i,col in enumerate(y_test.columns):
        # print(list(y_test[col]))
        # print(y_pred[:,i])
        df.loc[col,"accuracy"] =\
            m.accuracy_score(list(y_test[col]),y_pred[:,i])
    return df

def clf_fit_predict(clf, plot=True):
    """ fit, predict model and return accuracy_by_tag
    input: clf, plot=True
    output: accuracy_by_tag df
    """
    clf.fit(m.X_train, m.y_train)
    y_pred = clf.predict(m.X_test)
    df = accuracy_by_tag(m.y_test, y_pred)
    # print(hamming_loss(m.y_test, y_pred))
    df.sort_values(by="accuracy",inplace=True)
    if plot:
        df.plot(kind="bar",figsize=(10,4),title=clf)
    return df