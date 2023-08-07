def print_subdirs(save_path, **kwargs):
    print(kwargs)
    return save_path

pred_path = print_subdirs(save_path=None, sub_dirs="confusion")

# using asterisk
def add(hi,*per):
  print("alo hi", hi)
  return print(per)
add("hi",1,2,"confusion",4)



def create_pred_path(save_path, aimed=False, weighted=False,**kwargs):
    value = next(iter(kwargs.values()))
    print(value)
    return 

create_pred_path(save_path=None, aimed=False, weighted=False,s="t")


