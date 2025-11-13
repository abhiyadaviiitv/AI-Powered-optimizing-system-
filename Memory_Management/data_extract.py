import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

class PageAccessSimulator:
    """
    Simulates page access patterns for memory management training data
    """
    def __init__(self, num_pages=100, memory_size=20):
        self.num_pages = num_pages
        self.memory_size = memory_size
        self.access_history = []
        
    def generate_locality_based_access(self, num_accesses=10000):
        """
        Generate page accesses with temporal and spatial locality
        """
        current_page = random.randint(0, self.num_pages - 1)
        
        for _ in range(num_accesses):
            # 70% temporal locality (recently accessed pages)
            if random.random() < 0.7 and len(self.access_history) > 0:
                # Access from recent pages
                recent_pages = [entry['page_id'] for entry in self.access_history[-50:]]
                current_page = random.choice(recent_pages)
            # 20% spatial locality (nearby pages)
            elif random.random() < 0.2:
                offset = random.randint(-5, 5)
                current_page = max(0, min(self.num_pages - 1, current_page + offset))
            # 10% random access
            else:
                current_page = random.randint(0, self.num_pages - 1)
            
            yield current_page
    
    def extract_features(self, page_id, memory_state, timestamp):
        """
        Extract features for each page in memory
        """
        features = []
        
        for mem_page_id in memory_state:
            # Calculate recency (time since last access)
            last_access_times = [entry['timestamp'] for entry in self.access_history 
                               if entry['page_id'] == mem_page_id]
            recency = len(self.access_history) - max([i for i, entry in enumerate(self.access_history) 
                                                       if entry['page_id'] == mem_page_id], default=0)
            
            # Calculate frequency (number of accesses)
            frequency = len([entry for entry in self.access_history 
                           if entry['page_id'] == mem_page_id])
            
            # Time in memory
            time_in_memory = len([entry for entry in self.access_history[-100:] 
                                 if entry['page_id'] == mem_page_id])
            
            # Distance from current access
            page_distance = abs(mem_page_id - page_id)
            
            # Recent access pattern (accessed in last N accesses)
            recent_10 = 1 if mem_page_id in [e['page_id'] for e in self.access_history[-10:]] else 0
            recent_50 = 1 if mem_page_id in [e['page_id'] for e in self.access_history[-50:]] else 0
            recent_100 = 1 if mem_page_id in [e['page_id'] for e in self.access_history[-100:]] else 0
            
            features.append({
                'page_id': mem_page_id,
                'recency': recency,
                'frequency': frequency,
                'time_in_memory': time_in_memory,
                'page_distance': page_distance,
                'recent_10': recent_10,
                'recent_50': recent_50,
                'recent_100': recent_100,
                'timestamp': timestamp
            })
        
        return features
    
    def simulate_and_extract(self, num_accesses=10000, output_file='page_access_data.csv'):
        """
        Simulate page accesses and extract training data
        """
        memory = []
        data_records = []
        page_faults = 0
        
        print(f"Starting simulation with {num_accesses} accesses...")
        
        for i, page_id in enumerate(self.generate_locality_based_access(num_accesses)):
            timestamp = i
            
            # Record access
            self.access_history.append({
                'page_id': page_id,
                'timestamp': timestamp
            })
            
            # Check if page fault
            if page_id not in memory:
                page_faults += 1
                
                # If memory is full, we need to evict
                if len(memory) >= self.memory_size:
                    # Extract features for all pages in memory
                    features = self.extract_features(page_id, memory, timestamp)
                    
                    # Calculate optimal victim using Belady's algorithm (future knowledge)
                    victim_page = self.find_optimal_victim(page_id, memory)
                    
                    # Mark the victim page
                    for feature in features:
                        feature['is_victim'] = 1 if feature['page_id'] == victim_page else 0
                        feature['incoming_page'] = page_id
                        data_records.append(feature)
                    
                    # Evict victim
                    memory.remove(victim_page)
                
                # Add new page to memory
                memory.append(page_id)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_accesses} accesses, Page faults: {page_faults}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data_records)
        df.to_csv(output_file, index=False)
        
        print(f"\nSimulation complete!")
        print(f"Total accesses: {num_accesses}")
        print(f"Total page faults: {page_faults}")
        print(f"Page fault rate: {page_faults/num_accesses*100:.2f}%")
        print(f"Training samples generated: {len(data_records)}")
        print(f"Data saved to: {output_file}")
        
        return df
    
    def find_optimal_victim(self, incoming_page, memory):
        """
        Belady's optimal algorithm - finds page that will be used furthest in future
        Uses lookahead to find the best victim
        """
        future_accesses = {}
        
        for page in memory:
            # Look ahead in access pattern
            future_distance = float('inf')
            for i in range(len(self.access_history), min(len(self.access_history) + 1000, len(self.access_history) + 1000)):
                # Simulate future accesses (we can use pattern prediction here)
                # For simplicity, using a heuristic based on past pattern
                pass
            
            # Find when this page will be accessed next (approximation)
            recent_accesses = [entry['page_id'] for entry in self.access_history[-100:]]
            if page not in recent_accesses:
                future_distance = 1000
            else:
                future_distance = 100 - recent_accesses[::-1].index(page)
            
            future_accesses[page] = future_distance
        
        # Return page with maximum future distance (used furthest in future)
        victim = max(future_accesses.items(), key=lambda x: x[1])[0]
        return victim

if __name__ == "__main__":
    # Create simulator
    simulator = PageAccessSimulator(num_pages=100, memory_size=20)
    
    # Generate and extract data
    df = simulator.simulate_and_extract(num_accesses=10000, output_file='page_access_data.csv')
    
    print("\nSample data:")
    print(df.head(10))
    print("\nFeature statistics:")
    print(df.describe())