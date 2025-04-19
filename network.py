from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from multiprocessing.pool import ThreadPool
import hashlib
import pathlib
from openai import OpenAI

CLIENT = OpenAI()

# Create cache directory if it doesn't exist
cache_dir = pathlib.Path('.cache')
cache_dir.mkdir(exist_ok=True)
(cache_dir / "bluesky").mkdir(exist_ok=True)
(cache_dir / "openai").mkdir(exist_ok=True)

from api import PublicBlueskyAPI


def process_handle(args):
    """Helper function to process a single handle with caching"""
    api: PublicBlueskyAPI
    api, handle, limit = args
    
    # Create cache key from handle and limit
    cache_key = f"{handle}_{limit}"
    cache_file = cache_dir / "bluesky" / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"
    
    # Check cache first
    if cache_file.exists():
        with open(cache_file) as f:
            return handle, json.load(f)
    
    # If not in cache, make API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            follows = api.get_follows(handle, limit=limit)
            posts = api.get_posts(handle, limit=10)
            profile = api.get_profile(handle)
            data = {
                'follows': follows,
                'profile': profile,
                'recent_posts': posts
            }
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            return handle, data
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get data for {handle} after {max_retries} attempts: {e}")
                return handle, None
            print(f"Attempt {attempt + 1} failed for {handle}: {e}. Retrying...")

def get_network(api, start_handle, depth=2, limit=100):
    """Build network of follows starting from given handle up to specified depth"""
    G = nx.DiGraph()
    handles_to_process = {start_handle}
    processed_handles = set()
    failed_handles = set()
    
    # Store additional node data
    node_data = {}
    
    # Create thread pool
    pool = ThreadPool(processes=10) # dies with more than 10
    
    for _ in range(depth):
        current_handles = handles_to_process.copy()
        handles_to_process = set()
        
        # Create args list for pool
        args_list = [(api, handle, limit) for handle in current_handles 
                    if handle not in processed_handles and handle not in failed_handles]
        
        # Process handles in parallel
        for handle, data in tqdm(
            pool.imap_unordered(process_handle, args_list),
            total=len(args_list),
            desc=f"Processing depth {_ + 1}/{depth}"
        ):
            if not data:
                failed_handles.add(handle)
                continue
                
            # Store node data
            node_data[handle] = {
                'profile': data['profile'],
                'recent_posts': data['recent_posts']
            }
            
            # Add edges from this user to all their follows
            for follow in data['follows']:
                follow_handle = follow['handle']
                G.add_edge(handle, follow_handle)
                handles_to_process.add(follow_handle)
                
            processed_handles.add(handle)
            
    pool.close()
    pool.join()
    
    if failed_handles:
        print(f"Failed to process {len(failed_handles)} handles: {failed_handles}")
    
    return G, node_data

def save_network(G, node_data, filename):
    """Save network data to JSON file"""
    data = {
        'nodes': {node: node_data.get(node, {}) for node in G.nodes()},
        'edges': list(G.edges())
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_network(filename):
    """Load network data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'].keys())
    G.add_edges_from(data['edges'])
    return G, data['nodes']

def plot_network(G, output_file):
    """Plot and save the network visualization"""
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, 
            node_color='lightblue',
            node_size=500,
            with_labels=True,
            font_size=8,
            arrows=True,
            edge_color='gray',
            alpha=0.6)
    plt.savefig(output_file)
    plt.close()


def clean_data(node_data):
    # Save structured profile data
    structured_data = {}
    for handle, data in node_data.items():
        # Add profile data
        profile = data.get('profile')
        if profile:
            clean_data = {
                'display_name': profile.get('display_name'),
                'description': profile.get('description'),
                'followers_count': profile.get('followers_count'),
                'follows_count': profile.get('follows_count'),
                'posts_count': profile.get('posts_count'),
                'handle': profile.get('handle'),
                'did': profile.get('did')
            }

        # Add ChatGPT classification
        clean_data['of_interest'] = data['of_interest']
        
        # Add recent posts
        clean_data['recent_posts'] = []
        if 'recent_posts' in data:
            for post in data['recent_posts']:
                post_data = {
                    'author': {
                        'did': post['post']['author'].get('did'),
                        'handle': post['post']['author'].get('handle'),
                        'display_name': post['post']['author'].get('display_name')
                    },
                    'text': post['post']['record'].get('text'),
                    'like_count': post['post'].get('like_count'),
                    'quote_count': post['post'].get('quote_count'),
                    'reply_count': post['post'].get('reply_count'),
                    'repost_count': post['post'].get('repost_count')
                }
                clean_data['recent_posts'].append(post_data)
        
        structured_data[handle] = clean_data
    
    return structured_data


def _classify_node_prompt(unstructured_node_data, max_posts=10):
    prompt = "Based on this Bluesky user's profile information, assess whether they likely work on large language models (LLMs), natural language processing (NLP), or machine learning (ML). Consider their description, recent posts, and overall profile. Respond with just 'Yes' or 'No'.\n\n"

    profile = unstructured_node_data.get('profile')
    if profile:
        prompt += "Profile Information:\n"
        prompt += f"Display Name: {profile.get('display_name', '')}\n"
        prompt += f"Description: {profile.get('description', '')}\n"
    
    posts = unstructured_node_data.get('recent_posts', [])[:max_posts]
    if len(posts) > 0:
        prompt += "\nRecent Posts:\n"
        for post in posts:
            prompt += f"- {post['post']['record'].get('text', '')}\n"
    
    prompt += "\nDoes this person likely work on LLMs/NLP/ML? Please respond with just Yes or No."
    
    return prompt


def classify_node(node_data):
    # Generate prompt and create cache key
    prompt = _classify_node_prompt(node_data)
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = cache_dir / "openai" / f"{cache_key}.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)['result']
    
    response = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    result = response.choices[0].message.content.strip().lower() == 'yes'
    
    # Cache the result
    with open(cache_file, 'w') as f:
        json.dump({'result': result}, f)
        
    return result


def classify_data(unstructured_data):
    # Classify with OpenAPI
    def classify_node_wrapper(item):
        handle, data = item
        classification = classify_node(data)
        return handle, classification

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(classify_node_wrapper, item) for item in unstructured_data.items()]
        for future in tqdm(futures, desc='Classifying with OpenAI'):
            handle, classification = future.result()
            unstructured_data[handle]['of_interest'] = classification

    return unstructured_data

def main():
    load_dotenv()

    api = PublicBlueskyAPI()

    handle = "nlpnoah.bsky.social"
    print(f"Building network starting from {handle}...")

    network, unstructured_data = get_network(api, handle, depth=2, limit=1000)

    unstructured_data = classify_data(unstructured_data)

    interesting_children = [h for h, d in unstructured_data.items() if d.get('of_interest')]

    for child_handle in interesting_children:
        print(f"Exploring child node {child_handle}...")

        network, child_unstructured_data = get_network(api, child_handle, depth=2, limit=1000)
        child_unstructured_data = classify_data(child_unstructured_data)
    
        # Add new data from child networks if handle not already present
        for handle, data in child_unstructured_data.items():
            if handle not in unstructured_data:
                unstructured_data[handle] = data

        # Print statistics about the data
        total_entries = len(unstructured_data)
        interesting_entries = len([h for h, d in unstructured_data.items() if d.get('of_interest')])
        print(f"Total entries: {total_entries}")
        print(f"Interesting entries: {interesting_entries}")

        with open("viz/unstructured_profiles.json", "w") as f:
            json.dump(unstructured_data, f, indent=2)

        structured_data = clean_data(unstructured_data)

        # Only keep the entries where of_interest is True
        structured_data = {handle: data for handle, data in structured_data.items() if data.get('of_interest')}

        with open("viz/structured_profiles.json", "w") as f:
            json.dump(structured_data, f, indent=2)
    
    # # Save network data
    # network_file = "viz/bluesky_network.json"
    # print(f"\nSaving network data to {network_file}...")
    # save_network(network, node_data, network_file)
    
    # print(f"\nNetwork statistics:")
    # print(f"Number of nodes: {network.number_of_nodes()}")
    # print(f"Number of edges: {network.number_of_edges()}")
    
    # # Save network visualization
    # output_file = "bluesky_network.png"
    # print(f"\nSaving network visualization to {output_file}...")
    # plot_network(network, output_file)
    # print("Done!")

if __name__ == "__main__":
    main()
