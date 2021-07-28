---
title: Who's That Multimon?! Part 1 - Sprites and Dataloaders
categories: [Projects, Multimon]
tags: [data, neural networks, computer vision, pytorch]
published: true
layout: post
comments: true
pin: true
---

# Introduction
A friend recently suggested to me it would be a fun challenge to see if you could build a neural network to identify Pokémon types using just their sprites. Given I was in the process of trying to get my head round neural networks and learning PyTorch at the time this seemed like a fun little exercise to ease me into it! These posts will tell the story of how I got on, they'll ideally bridge the gap between being comprehendable and having just enough of the gory details for those more interested in that side. If you just want the technical stuff you can find the GitHub repo of my results [here](https://github.com/ImperialSquid/Multimon), without further ado, let's begin!

# Why the Project is Called Multimon
So first of all, why that name for the project? Since at the time of writing I am doing a PhD on the topic of Multitasking Neural Networks (ie a neural net where we have more than one output), I decided to expand on the original pitch by my friend to include a few more things. As opposed to just the type I wanted to get: 
- It's Type (such as Flying, Grass, etc)
- The Generation it's from (pokemon games are released in generations, the intuition here being that the design principles may have changed over time)
- Whether or not it's a "Shiny" (shiny pokemon is the term given to a rarer form of each pokémon, they are identical in all regards except that they have a different colout palette)

# The Data
The most important thing now therefore is to get some data to work on! for this I landed on using the Pokébase library for python ([GitHub here](https://github.com/PokeAPI/pokebase)) which acts as a lightweight interface for the PokéAPI database.










What data
Image cleaning
Data loader
Dataset