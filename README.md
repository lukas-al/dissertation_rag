# structured_rag
Dissertation project, MSc Data Science 2024.

This version in Github is out of date and in cold storage. The code has since moved to the internal repo.

---

## Narrative to construct

### Having:
1. Tested numerous embedding models
	- miniLMV12
	- multiqa-mpnet
	- BAAI/bge-large-en-v1.5

By manually scoring on some dataset of how similar we'd expect certain passages to be.

2. Improved the pre-processing of the text
	- Raw text
	- Basic filtering

Is there a way we can compare performance here?

3. Tested multiple AG LLMs
	- FLAN
	- BGE / LLama2 / Mixtral flavours

Is there a way we can compare performance here?

### We then compared various RAG systems:
1. Simple RAG LLM without Graph context
2. LLM with Graph - embeddings only
3. LLM with Graph - different methods 0-4
 
### We scored the different RAG systems using:
1. LLM-generated Question Answer couples, combining context with prompts.
	- These are critiqued by another LLM to filter out bad questions and bad contexts.
    - Then scored using another evaluation LLM to see how the RAG system performed.

2. A manually constructed Question set with some needle-in-haystack queries for the LLM.
    - Manually scored by 2 different individuals

### Ultimately, we came to some conclusions on the uniqueness of this dataset and the performance of the RAG systems.
1. TBD



-----

# Old todo
- Get a narrative going
    - Bring in data from a bunch of different time periods
    - Construct examples of performance differences

- Search is somewhat ok as you can filter
- RAG is worse as it leads to confusion

- Stuff the document with the metadata

To do detail:
- Split the documents into appropriately sized nodes 
    - Make sure the algo and all works the same once docs are nodes
    - This is so that the QA LLM can parse the nodes rather than cutting them off.
- Figure out how to fit the metadata into the context window with the node splitter
- Modify the ONS code to have a different Pydantic parser layout so that we actually get useful responses from the LLM.
    - Need to understand the pydantic parser class for this


## Overview
See the project initiation document if you're interested...

## Next Steps
1. Get a RAG working on a bunch of test data.
2. Select the data to use for the demo.
2. Build some test cases and evaluation criteria for the RAG system.
3. Iteratively improve the system, checking against the evaluation criteria and test results.

## Selecting the data to use

#### Considerations
- Version 1: Only using external publications
- Version 2: Using also internal publications (notes portal) - This would be much more useful.

At first instance, we're going to do Version 1 (external publications on webiste).
- I could do MPR to MPR, but the external publications don't map so well to that.
- Instead, I'm just going to do a full quarter.
- The date to use could be `2023 March - June` (incl. the MPR)

```HAVE BASICALLY LOADED EVERYTHING IN JANUARY 2020```

What about which content to use?
| Type | Notes | Included? | N |
| - | - | - | - |
| Research Blog | Very rare - few of these exist on the website | ✅ | 0 |
| Event | Typically don't have a lot of text or info to extract | ❌ | 14 |
| Explainers | Rare form of content with custom formats | ❌ | 2 |
| News | Can be easily converted to PDF. A lot of random content | ✅ | 48 |
| Prudential Regulation | Lets include it as well | ✅ | 20 |
| Publication | Yes | ✅ | 45 | 
| Speech | Semantically rich | ✅ | 24 |
| Statistics | Interesting to see how these are used | ✅ | 21 |
| - | - | Total | 158 |

Looking at the landscape of available content makes me think that I'll need to use the Notes portal.  I don't think there is enough overlap in a lot of the data from the external publications and it doesn't combine well. On the other hand, the Notes portal has a lot of overlapping content and much more metadata to create other structural links.


## Ideas for more matching:
Linguistic and structural features that help us to match the tree
- Matching numbers
- Pairs or triples of words
- Named entity recognition
- Links

Authorship, year, etc are obviously going to turbocharge the document recognition. But can other linguistic features work as well? This is a useful question.

Essentially:
- You could get a really good search based on metadata, named entity recognition, etc
- How good can it get without the embeddings?
- And vice-versa

This should result in a big table of scores.

Interesting question:
- To what level is matching numbers and named entities a good proxy for date

Another Idea:
- Overloading the embedding with metadata - put it all into the document text and embed that entirely at once.
- Does this deal with blanks better? Does it deal with nulls better.

---
