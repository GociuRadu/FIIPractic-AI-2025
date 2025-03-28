class Node :
    def __init__(self,column,value=None,true_branch=None,false_branch=None,label=None):
        self.column=column
        self.value=value
        self.true_branch=true_branch
        self.false_branch=false_branch
        self.label=label

def is_leaf :
    return self.lable is not None