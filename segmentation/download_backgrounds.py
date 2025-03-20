import os
def download_backgrounds(output_dir, num_images=300):
    """Download background images automatically from search engines or APIs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # List of search terms for diverse backgrounds 
    search_terms = ["مجلس خليجي", "قهوة عربية مجلس", "صحن تمر شعبي", "جلسات قهوة عربية", "صحن نواة التمر", "صحن عبس التمر", "قهوة عربية تمر", "human hand open", "مزرعة تمور"]
    
  
    from bing_image_downloader import downloader
    
    images_per_term = num_images // len(search_terms) + 1
    
    for term in search_terms:
        # Downloads images to output_dir/term/
        downloader.download(
            term, 
            limit=images_per_term,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )
    
  
    
    return output_dir

if __name__ == "__main__":
    download_backgrounds("backgrounds")