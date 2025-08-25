# %%
# genetic algorithm search of the one max optimization problem
import numpy as np
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import math
import cv2
import os

# %% [markdown]
# ## Objective function

# %%
def color_match(x, target):
	dist = 0
	for c1, c2 in zip(x, target):
		dist += (c1-c2)**2
	return math.sqrt(dist)
if 0:
	def color_match(l_pop, target):
		err = 0
		for clr in l_pop:
			err += math.sqrt((clr[0]-target[0])**2 + (clr[1]-target[1])**2 + (clr[2]-target[2])**2)
		return err

# %% [markdown]
# ## Elite+Roulette selection

# %%
import pandas as pd

def scores2prob(scores):
    inv_scores = [ 1/(score+ 1e-10) for score in scores]
    agg = sum(inv_scores)
    prob = [ score/agg for score in inv_scores]
    
    return prob

def selection(pop, scores):
    
    nsamples = len(pop)
    df = pd.DataFrame({
          'pop': pop,
          'scores': scores
          })
    
    df = df.sort_values(by=['scores']).reset_index(drop=True) # sorted in ascending order
    new_parents = df['pop'] #.tolist()[:top_k]
    new_scores = df['scores'] #.tolist()[:top_k]
    prob = scores2prob(new_scores)
    
    sample_idx = [np.random.choice(range(len(new_parents)), 2, replace=False, p=prob).tolist() for _ in range(nsamples)]

    ldict_selection = []
    for p1_idx, p2_idx in sample_idx:
        agg = prob[p1_idx]+prob[p2_idx]
        d_sel = {
            'parents': [new_parents[p1_idx], new_parents[p2_idx]],
            'scores': [prob[p1_idx]/agg, prob[p2_idx]/agg]
            }
        ldict_selection.append(d_sel)
	
    return ldict_selection


# %% [markdown]
# ## Arithmetic crossover

# %%
def crossover(ldict_selection):
	l_child = []
	for d_sel in ldict_selection:
		p1 ,p2 = d_sel['parents']
		s1, s2 = d_sel['scores']
		child = (s1 * np.array(p1) + s2 * np.array(p2)).astype(np.uint8).tolist()

		l_child.append(child)
	
	return l_child

# %% [markdown]
# ## Mutation operator

# %%
from copy import deepcopy

def mutation(l_child, r_mut):
	l_mutation = deepcopy(l_child)
	for sidx, sample in enumerate(l_mutation):
		for cidx, clr in enumerate(sample):
			# check for a mutation
			if rand() < r_mut:
				mask = np.uint8(1 << randint(0, 8))
				sample[cidx] = clr ^ mask
		l_mutation[sidx] = sample
	return l_mutation


# %% [markdown]
# ## Display the interation results

# %%
def display_pallete(sample_pop, target_color, 
                    children, mutations, best, generation,
                    n_row, n_col):
    pop_palette = np.array(sample_pop)[np.newaxis, :, :].reshape(n_row, n_col, 3)
    chd_palette = np.array(children)[np.newaxis, :, :].reshape(n_row, n_col, 3)
    mut_palette = np.array(mutations)[np.newaxis, :, :].reshape(n_row, n_col, 3)
    
    target_pal = np.array([target_color])[np.newaxis, :, :]
    best_pal = np.array([best])[np.newaxis, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    ax[0].imshow(target_pal)
    ax[0].axis('off')
    ax[0].title.set_text('Target')

    ax[1].imshow(pop_palette)
    ax[1].axis('off')
    ax[1].title.set_text(f'Generation:{str(generation).zfill(3)} parents')

    ax[2].imshow(chd_palette)
    ax[2].axis('off')
    ax[2].title.set_text(f'Generation:{str(generation).zfill(3)} children')

    ax[3].imshow(mut_palette)
    ax[3].axis('off')
    ax[3].title.set_text(f'Generation:{str(generation).zfill(3)} mutations')

    ax[4].imshow(best_pal)
    ax[4].axis('off')
    ax[4].title.set_text(f'Generation:{str(generation).zfill(3)} best result')

    plt.tight_layout()
    plt.savefig(f'./output/results_{str(generation).zfill(3)}.png')
    plt.close()


def createVideo():
	
    image_folder = 'output'
    video_name = './output/video.avi'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	
    images.sort()
	
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
	
    cv2.destroyAllWindows()
    
    video.release()


# %% [markdown]
# ## Genetic algorithm

# %%
def genetic_algorithm(pop, target,
					  objective, n_iter, r_mut, d_disp_grid_sz):
	
	n_pop = len(pop)
	# keep track of best solution
	best, best_eval = pop[0], objective(pop[0], target)
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c, target) for c in pop]
		
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		
		# select parents
		selected = selection(pop, scores)
		# create the next generation
		children = crossover(selected)
		mutations = mutation(children, r_mut)

		# Display results from the generation
		display_pallete(pop, target, children, mutations, best, gen,
				  d_disp_grid_sz['N_ROW'], d_disp_grid_sz['N_COL'])

		# replace population
		pop = mutations
	
	createVideo()
	
	return [best, best_eval]


# %% [markdown]
# ## Create initial population (color pallete in this case)

# %%
N_ROW, N_COL = 5, 10
N_POPULATION = N_ROW * N_COL
sample_pop = [tuple(randint(128, 255, 3)) for _ in range(N_POPULATION)]
target_color = (0, 0, 0) #tuple(randint(0, 255, 3))

n_iter = 150
r_mut = 0.1
d_disp_grid_sz = {
    'N_ROW' : N_ROW,
    'N_COL' : N_COL
}

if not os.path.exists('./output/'):
    os.makedirs('./output/')

genetic_algorithm(sample_pop, target_color,
					  color_match, n_iter, r_mut, d_disp_grid_sz)

# %%



