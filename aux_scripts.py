import random
import string
import os

def generate_id():
    my_set = set(i for i in string.ascii_uppercase)
    [my_set.add(i) for i in string.digits]
    id_cap = random.choices(list(my_set), k=8)
    id_cap = ''.join(id_cap)
    return id_cap


def change_name_of_files():
    folder = "./img"
    for f in os.listdir(folder):
        source = folder + "/" + f
        destination = folder + "/" + generate_id() + ".jpg"
        os.rename(source, destination)