---
title: Who's That Multimon?! Part 1 - Sprites and Dataloaders
categories: [Projects, Multimon]
tags: [neural networks, multitask learning, computer vision, pytorch, pokémon]
published: true
layout: post
comments: false
pin: true
toc: true
image:
  src: /assets/headers/multimonheader.png
  width: 1100   # in pixels
  height: 118   # in pixels
  alt: Multimon header image, says Who's That Multimon?! in pokémon style font with a Charizard sprite
---

# Introduction
A friend recently suggested it would be a fun challenge to see if you could build a neural network to identify Pokémon types using just their sprites. Given I was in the process of trying to get my head around neural networks and learning PyTorch at the time, this seemed like a fun little exercise to ease me into it! These posts will tell the story of how I got on. They’ll ideally bridge the gap between being comprehendible and having just enough of the gory details for those more interested in that side. If you want the technical stuff, you can find the GitHub repo of my results [here](https://github.com/ImperialSquid/Multimon). Without further ado, let’s begin!


# Why is the Project called "Multimon"?
So, first of all, why that name for the project? Since, at the time of writing, I am doing a PhD on Multitasking Neural Networks (i.e. a neural net where we have more than one output), I decided to expand on the original pitch by my friend to include a few more things. As opposed to just the type I wanted to get: 
![Lucario](/assets/multimon/448.png){: .right width="240" height="300" }
- It's Type (such as Flying, Grass, etc)
- The Generation it's from (pokémon games are released in generations, the intuition here being that the design principles may have changed over time)
- Whether or not it's a "Shiny" (shiny pokémon is the term given to a rarer form of each pokémon, they are identical in all regards except that they have a different colout palette)

So for example, with the Lucario sprite shown to the right, we want to guess Fighting and Steel for the types, Gen IV for the generation and False for the shininess!


# The Data
The most important thing now, therefore, is to get some data to work on! I landed on using the Pokébase library for python ([GitHub here](https://github.com/PokeAPI/pokebase)), which acts as a lightweight interface for the PokéAPI database.

So for each sprite, I also need to store the correct types, generation and shininess. Given that for each pokémon, I want multiple sprites (for shiny/not shiny and also for front/back sprites), it’s going to be easier, in the long run, to collect the target data for each pokémon in advance. Then access it as we download the sprites, rather than collecting the data each time.


## The Types
While it may seem easier to collect the types for each pokémon one by one, this ends up being **far** slower than collecting all the pokémon of each type and *then* flipping it the other way since there are less than two dozen types and several hundred pokémon!

To acheive this, I keep getting the pokemon for each type until I get an error, then using the pokemon in the list for that type, I make a dictionary mapping pokemon to their types. The pseudocode for this is below:

```
type_id = 0
type_index = dict() 
# will eventualy form a dictionary that looks like: 
# {poke1: [type11, type12], poke2: [type21, type22], ...}
WHILE TRUE:
    TRY:
        type = GET type name using pokebase
        pokemons = GET all pokemon with that type
    EXCEPT type doesn't exist:
        BREAK
    type_id += 1
    
    FOR pokemon IN pokemons:
        old_types = type_index[pokemon] or [] if it dowsn't exist
        new_types = old_types + [type]
        # new_types is now a list of the types that pokemon has so far
        type_index[pokemon] = new_types
    ENDFOR
ENDWHILE
```

For reasons I'll explain when I talk about the model in the next part (link to be added) I also added a bit of code to add a second "none" type if the pokémon only has one type.

## The Generation
Getting the generation for each pokémon works much the same as the above portion of code for types. I also keep the same pokemon to generation mapping format as above to make adding everything together later more straightforward.

## The Shininess
We don’t actually need to collect this in advance! Since *every* pokémon has a shiny and non-shiny version, we can construct this target bit of data on the fly as we download the sprites!

## The Sprites
Now for the last thing: downloading the sprites! I intend just to download the sprites raw initially, and then I don’t need to re-download them if I do some extra preprocessing later. I use the same idea as the types and generations previously to keep downloading sprites until I run out. I also need to download different sprite versions (shiny/not shiny and front/back). To do this, I download every pokémon of one version, then all of the next and so on. The pseudocode for this is as follows:

```
FOR kwargs = {every combo of True/False for "shiny" and "back"}
    suffix = each "True" kwarg with an underscore between
    # eg {"shiny": True, "back": True} -> "shiny_back"
    # but {"shiny": False, "back": True} -> "back"
    
    index = 1
    WHILE TRUE:
        TRY:
            img_data = GET sprite with ID index
        EXCEPT sprite doesn't exist:
            BREAK
        index += 1
        
        filename = index + suffix + ".jpg"
        filepath = "./sprites/raw/" + filename
        IF filename doesn't exist:
            WRITE img_data TO filepath
```

I also did a bit of preprocessing to standardise the images. Since the size of the images varied between 96x96 and 128x128 pixels, I rescaled everything down to 96x96. I also removed any transparency and gave the sprites a black background.

## Connecting It All Together
With the sprites downloaded and preprocessed, and all the target data for each pokémon downloaded, I now needed a way to tie them together. I did this by taking a list of all the sprites, extracting the ID from the name, and getting the relevant types, generations, and shininess. This is done as follows:

```
WITH open("data.txt") AS file:
    FOR image_name IN image_folder:
        id = extract id from image_name
        # shiny pokémon have the "shiny" in their name as per The Sprites
        shininess = extract shininess from image_name
        types = type_index[id][0] + "," + type_index[id][1]
        gen = gen_index[id]
        
        text = image_name + "," + types + "," + gen + "," + shininess
        WRITE text TO file
```

# The Dataset
The last thing in this post will be making a proper Dataset object from PyTorch. PyTorch uses Dataset objects to handle loading the data as training happens, they can be about as simple or complex as you need. Still, the minimum you need is a method to dispense the data and another to return how many data items there will be (so PyTorch knows not to request too many instances).

The first thing to do is make a way to get the data back out of the `data.txt` file we made previously. This isn’t hard to do, but more work will need to be done on the target data after this is sorted. Having a pokémon’s type be “fire” or “water” doesn’t mean much to a neural net since that only works with numbers. I, therefore, turned these categories into “one-hot vectors” (or two-hot in the type scenario).

A one-hot vector encodes each category to an index in a target vector. The corresponding element is then “hot” and marked with a 1, while the rest are zeroes. The model then tries to get the correct element as close to 1 and as close to 0 for the others.

```
An Example One-Hot Encoding
Dog -> [1,0,0]
Cat -> [0,1,0]
Fish -> [0,0,1]
```

To parse the data file, I then convert the text file like so (to those that know the difference, the data is also converted to tensors as it gets parsed, if not, don’t worry about it, we’ll get more into PyTorch specifics later!):

```
data_dict = dict()
FOR line IN text_file:
    splits = split lines along ","s
    index = splits[0]
    
    types = EncodeOneHot(splits[1]) + EncodeOneHot(splits[2])
    # "fire,electric" -> [1,0,0,...] + [0,0,1,...] = [1,0,1,...]  
    
    gen = EncodeOneHot(splits[3])
    # "gen-3" -> [0,0,1,...]
    
    shiny = (splits[4] == "true")
    # "false" -> 0
    
    data_dict[index] = [types, gen, shiny]
```

In machine learning, it’s best to keep separate data sets for training and testing the model. Since the data is mixed together, I also needed to filter it as it gets loaded to keep separate sets. This is achieved with some python [dictionary comprehension](https://www.datacamp.com/community/tutorials/python-dictionary-comprehension). 

```
full_data = get full data dictionary from data.txt
filtered_data = {key: full_data[key] for index, key in enumerate(full_data) 
                 if index in index_mask or no_mask}
```

This line may look a bit complicated... Essentially, all it does is take in a list of indexes to keep (e.g. [1,2,4,5,7,8,…]) and throws out any data element not in the index mask. Alternatively, if no_mask is True, all elements are added regardless of the index mask.

I plan to do several other fancy things with this Dataset later, but that is all that’s needed for now!

Here’s a chunk of pseudocode for the Dataset (with lots of little things skimmed over that I’ll come back to later)

```
class MultimonDataset(Dataset):
    self __init__(self, data_file, img_path, index_mask, no_mask = False)
        full_data = self.parse_data_file(data_file)
        self.data = {key: full_data[key] for index, key in enumerate(full_data) 
		             if index in index_mask or no_mask}
        # filter data as before
        
        self.img_path = img_path
    
    self __len__(self)
        return len(self.data)
    
    self __getitem__(self, index):
        key = self.data.keys()[index]
        img = READ (self.img_path + key)
        labels = self.data[key]

    self parse_data_file(self, filename):
        # Parse data as described above
```

# Summary
Congrats! With that, we’ve come to the end of Part 1 of the project! I set out the rough outline of the project and got a lot of the data management sorted! Next up, I’ll very quickly skim over some machine learning basics and start making the model!
