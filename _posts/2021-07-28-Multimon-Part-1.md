---
title: Who's That Multimon?! Part 1 - Sprites and Dataloaders
categories: [Projects, Multimon]
tags: [neural networks, multitask learning, computer vision, pytorch, pokémon]
published: true
layout: post
comments: true
pin: true
toc: true
---

# Introduction
A friend recently suggested to me it would be a fun challenge to see if you could build a neural network to identify Pokémon types using just their sprites. Given I was in the process of trying to get my head round neural networks and learning PyTorch at the time this seemed like a fun little exercise to ease me into it! These posts will tell the story of how I got on, they'll ideally bridge the gap between being comprehendable and having just enough of the gory details for those more interested in that side. If you just want the technical stuff you can find the GitHub repo of my results [here](https://github.com/ImperialSquid/Multimon), without further ado, let's begin!

# Why the Project is Called Multimon?
So first of all, why that name for the project? Since at the time of writing I am doing a PhD on the topic of Multitasking Neural Networks (ie a neural net where we have more than one output), I decided to expand on the original pitch by my friend to include a few more things. As opposed to just the type I wanted to get: 
- It's Type (such as Flying, Grass, etc)
- The Generation it's from (pokémon games are released in generations, the intuition here being that the design principles may have changed over time)
- Whether or not it's a "Shiny" (shiny pokémon is the term given to a rarer form of each pokémon, they are identical in all regards except that they have a different colout palette)

# The Data
The most important thing now therefore is to get some data to work on! For this I landed on using the Pokébase library for python ([GitHub here](https://github.com/PokeAPI/pokebase)) which acts as a lightweight interface for the PokéAPI database.

So for each sprite I want to train on, I also need to store the correct types, generation and shininess. Given that for each pokémon I actually want multiple sprites (for shiny/not shiny and also for front/back sprites) it's going to be easier in the long run to collect the target data for each pokémon in advance and then store it as we download the sprites, rather than collecting it each time.

## The Types
While it may seem easier to collect the types for each pokémon one by one, this ends up being **far** slower than collecting all the pokémon of each type and *then* flipping it the other way since there are less than two dozen types and several hundred pokémon!

To acheive this, I keep getting the pokemon for each type until I get an error, then using the pokemon in the list for that type, I make a dictionary mapping pokemon to their types. The pseudo code for this is below:

```
type_id = 0
type_index = dict() # will eventualy form a dictionary that looks like: {poke1: [type11, type12], poke2: [type21, type22], ...}
WHILE TRUE:
	TRY:
		type = GET type name using pokebase
		pokemons = GET all pokemon with that type
	EXCEPT type doesn't exist:
		BREAK
	type_id += 1
	
	FOR pokemon IN pokemons:
		old_types = type_index[pokemon] OR [] IF type_index[pokemon] DOWSN'T EXIST
		new_types = old_types + [type]
		# new_types is now a list of the types that pokemon has
		type_index[pokemon] = new_types
	ENDFOR
ENDWHILE
```

For reasons I'll explain when I talk about the model in the next post (link to be added) I also added a bit of code to add a second "none" type if the pokémon only has one type.

## The Generation
Getting the generation for each pokémon works much the same as the above types portion of code, I keep the same pokemon to generation mapping format as above too to make adding everythng together later more simple.

## The Shininess
We don't actually need to collect this in advance, since *every* pokémon has a shiny and non-shiny version, we can construct this target bit of data on the fly as we download the sprites!

## The Sprites

# Dataset

# Data loader
