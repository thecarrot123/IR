# 1. Assessor and analyst work

## 1.0. Rating and criteria

Please [open this document](https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf)
and study chapters 13.0-13.4. Your task will be to assess the organic answers of search engines given the same query.

## 1.1. Explore the page

For the following search engines:
- https://duckduckgo.com/
- https://www.bing.com/
- https://ya.ru/
- https://www.google.com/

Perform the same query: "**How to get from Kazan to Voronezh**".

Discuss with your TA the following:
1. Which elements you may identify at SERP? Ads, snippets, blends from other sources, ...?
2. Where are organic results? How many of them are there?

## 1.2. Rate the results of the search engine

If there are many of you in the group, assess all search engines, otherwise choose 1 or 2. There should be no less than 5 of your for each search engine. Use the scale from the handbook, use 0..4 numerical equivalents for . 

Compute:
- average relevance and standard deviation.
- [Fleiss kappa score](https://en.wikipedia.org/wiki/Fleiss%27_kappa#Worked_example). Use [this implementation](https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.fleiss_kappa.html).
- [Kendall rank coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient). Use [this implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).

Discuss numerical results. Did you agree on the relevance? Did you agree on the rank? What is the difference?

# 1. Solution:

First of all we put our data in ranking_data array:

```python3
import numpy as np

ranking_data = np.array([
    [4, 4, 4, 3, 4, 2, 2, 1, 1, 0], 
    [4, 3, 4, 3, 3, 2, 1, 1, 1, 1], 
    [3, 4, 4, 4, 4, 3, 2, 1, 1, 1], 
    [4, 4, 4, 4, 3, 2, 2, 1, 1, 0],
    [4, 4, 4, 4, 3, 2, 2, 1, 1, 3]
])```

Then we calculate the averages ang standard deviations per item.

```python3
import numpy as np

def compute_mean_and_sigma(ranking_data):
    average_relevance = np.mean(ranking_data, axis=0)
    sigma = np.std(ranking_data, axis=0)
    return average_relevance, sigma

def print_relevance_stats(average_relevance, sigma):
    for i, (mean, std) in enumerate(zip(average_relevance, sigma)):
        print(f"relevance {i}: {mean:.2f} +- {std:.3f}")

average_relevance, sigma = compute_mean_and_sigma(ranking_data)
print_relevance_stats(average_relevance, sigma)
```
After that we calculate fleiss kappa score using `aggregate_raters` from `statsmodels` library:

```python3 
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

aggregate, categories = aggregate_raters(ranking_data.T)
print(f"aggregate: {aggregate}")
print(f"categories: {categories}")
print(f"kappa: {fleiss_kappa(aggregate)}")
```

And kendall rank score:

```python3
from scipy.stats import kendalltau

for a in ranking_data:
    for b in ranking_data:
        print(kendalltau(a, b))
```

# 2. Engineer work

You will create a bucket of URLs which are relevant for the query **"free cloud git"**. Then you will automate the search procedure using https://serpapi.com/, or https://developers.google.com/custom-search/v1/overview, or whatever.

Then you will compute MRR@10 and Precision@10.

# 2. Solution:

First of all we build our bucket:

```python3
relevant_bucket = [
    "github.com",
    "gitpod.io",
    "bitbucket.org",
    "gitlab.com",
    "source.cloud.google.com",
    "sourceforge.net",
    "aws.amazon.com/codecommit/",
    "launchpad.net",
]

query = "free git cloud"
```

Then check if our query is relevant or not:

```python3
def is_relevant(resp_url):
    for u in relevant_bucket:
        if u in resp_url:
            return True
    return False
```

After that we get our automated search from https://serpapi.com/:

```python3
import requests 
api_key = "5aff1ae53da3a991a97d770bf1991833ba30a97d68925ede4cb0003285c727ba"

url = f"https://serpapi.com/search.json?q={query}&hl=en&gl=us&google_domain=google.com&api_key={api_key}"
js = requests.get(url).json()
```

Lastly build our relevant list:

```python3
rels = []
for result in js["organic_results"]:
    rels.append(int(is_rel(result['link'])))

print(rels)
```
Now to compute MRR:

```python3
def mrr(list_of_lists, k=10):
    r = 0
    for l in list_of_lists:
        r += (1 / (k + 1)) if 1 not in l else 1 / (l.index(1) + 1)
        #print(r)
    return r / len(list_of_lists)

mrr([rels])
```
And to compute mean precision:

```python3
def mean_precision(list_of_lists, k=10):
    p = 0
    for l in list_of_lists:
        p += sum(l) / k
    return p / len(list_of_lists)

mean_precision([rels])
```
