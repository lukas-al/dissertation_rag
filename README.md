# structured_rag
Try and fix the knowledge management issues within MA and the Bank of England.
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

