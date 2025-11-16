import fitz  # PyMuPDF
from google import genai  # Google Gen AI SDK (unified SDK - google-genai package)
import os
import json
import math
import time
from collections import deque
from PIL import Image
import io

# --- Configuration ---
# Load the API key from environment variables for security
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable not set in .env.")
    print("Please set the variable in your .env file and run the script again.")
    exit()

try:
    # Initialize the client with the new unified SDK 
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"ERROR: Failed to initialize Google Gen AI client: {e}")
    print("Make sure you have installed: pip install google-genai")
    exit()

# Define output directories
OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# --- Rate Limiting Configuration ---
# Gemini API rate limit: 15 requests per minute (RPM)
GEMINI_RPM_LIMIT = 15
GEMINI_RATE_LIMIT_WINDOW = 60  # seconds
MIN_TIME_BETWEEN_REQUESTS = 60 / GEMINI_RPM_LIMIT  # ~4 seconds between requests

# Track request timestamps for rate limiting
request_timestamps = deque()


def wait_for_rate_limit():
    """
    Implements rate limiting to respect Gemini API's 15 RPM limit.
    Ensures we don't exceed 15 requests per minute.
    """
    current_time = time.time()
    
    # Remove timestamps older than the rate limit window (60 seconds)
    while request_timestamps and (current_time - request_timestamps[0]) > GEMINI_RATE_LIMIT_WINDOW:
        request_timestamps.popleft()
    
    # If we've reached the limit, wait until we can make another request
    if len(request_timestamps) >= GEMINI_RPM_LIMIT:
        oldest_request_time = request_timestamps[0]
        wait_time = GEMINI_RATE_LIMIT_WINDOW - (current_time - oldest_request_time) + 0.1  # Add small buffer
        if wait_time > 0:
            print(f"  - Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            # Clean up old timestamps after waiting
            current_time = time.time()
            while request_timestamps and (current_time - request_timestamps[0]) > GEMINI_RATE_LIMIT_WINDOW:
                request_timestamps.popleft()
    
    # Ensure minimum time between requests (smoother distribution)
    if request_timestamps:
        time_since_last_request = current_time - request_timestamps[-1]
        if time_since_last_request < MIN_TIME_BETWEEN_REQUESTS:
            wait_time = MIN_TIME_BETWEEN_REQUESTS - time_since_last_request
            if wait_time > 0:
                time.sleep(wait_time)
    
    # Record this request timestamp
    request_timestamps.append(time.time())


# --- Phase 1: PDF Decomposition ---
def extract_page_elements(page, page_num):
    """
    Extracts images and text blocks with their bounding boxes from a single PDF page.
    """
    images_on_page = []
    text_blocks = []

    # 1. Extract Images
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image bounding box
            img_bbox = page.get_image_bbox(img)
            
            # Save image and store its info
            img_filename = f"page_{page_num+1}_img_{img_index}.png"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            
            # Use Pillow to handle various image formats and save as PNG
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert CMYK and other non-RGB modes to RGB for PNG compatibility
            if image.mode == 'CMYK':
                # Convert CMYK to RGB (PNG doesn't support CMYK directly)
                image = image.convert('RGB')
            elif image.mode == 'P':
                # Palette mode: check if it has transparency, convert to RGBA if so
                if 'transparency' in image.info:
                    image = image.convert('RGBA')
                else:
                    image = image.convert('RGB')
            elif image.mode in ('LAB', 'HSV', 'YCbCr'):
                # Convert other color spaces to RGB
                image = image.convert('RGB')
            elif image.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                # Convert any other unsupported modes to RGB
                image = image.convert('RGB')
            
            image.save(img_path, "PNG")

            images_on_page.append({
                "path": img_path,
                "bbox": tuple(img_bbox)
            })
        except (ValueError, IOError) as e:
            # This can happen for small, invalid, or inline images. We can safely ignore them.
            print(f"  - Warning: Could not process image {img_index} on page {page_num+1}. Skipping. Error: {e}")


    # 2. Extract Text Blocks
    blocks = page.get_text("blocks")
    for b in blocks:
        # b is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
        text_blocks.append({
            "text": b[4].strip(),
            "bbox": (b[0], b[1], b[2], b[3])
        })

    return images_on_page, text_blocks


# --- Phase 2: Intelligent Content Structuring (Gemini) ---
def get_structured_data_from_gemini(text_content):
    """
    Sends the extracted text to the Gemini API and asks for structured JSON output.
    Uses the new unified Google Gen AI SDK.
    """
    prompt = f"""
    You are an expert data extraction AI specializing in product catalogues for architectural hardware and e-commerce catalogs.
    Analyze the following text from a single catalogue page and return a valid JSON object.
    Your task is to identify every individual product *variation* and extract its details into a structured format.
    A single product line might result in multiple variations if it has different prices for different finishes (e.g., Glossy, Satin, Antique).
    
    The JSON object should contain a single key "products", which is a list of all products found on the page.

    For each product in the list, extract:
    1.  "product_name": The main name or model of the product. Include the finish in the name if applicable (e.g., "ANUBHUTI (BRASS)", "Mortice Handle Stainless Steel Finish").
    2.  "skus": A list of all variations (SKUs) for that product.

    For each SKU in the "skus" list, extract:
    - "product_code": The unique code or SKU identifier for this specific variation (e.g., "MH-ANT-275", "ABC123").
    - "sku_description": The full descriptive text for the SKU (e.g., "MH Anubhuti (ANT)- 275mm CY").
    - "finish": The specific product finish or color for this variation (e.g., "ANT", "GLOSSY BLK", "SS", "Satin", "Gold PVD", "Antique"). If no finish is specified, use "Standard".
    - "price": The price as a numerical value only. Remove any currency symbols (Rs., $, €, etc.), commas, and other non-numeric characters. Extract only the number (e.g., if the text says "Rs. 4,577", extract 4577 as a number).
    - "packaging": The standard packaging information, if available (e.g., "6 Pair", "12 units", "1 box").

    Important extraction rules:
    - If a product row lists multiple prices under columns like 'Glossy', 'Satin', 'Antique', create a separate SKU object for each one that has a price.
    - If a product is presented in a table, extract each row as a separate SKU, ensuring each variation with a distinct price or finish gets its own entry.
    - Ignore page headers, footers, logos, page numbers, and general text that is not associated with a specific product.
    - Focus on identifying every individual product variation, even if they share the same base product name.
    - If no products are found on the page, return a JSON object with an empty "products" list: {{"products": []}}.
    - Ensure the output is a single, clean JSON object without any markdown formatting.

    Here is the text from the page:
    ---
    {text_content}
    ---
    """
    
    try:
        # Respect rate limits before making API call
        wait_for_rate_limit()
        
        # Use the new unified SDK API
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        # Clean up potential markdown formatting from the response
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"  - Error processing with Gemini or parsing JSON: {e}")
        return {"products": []}


# --- Phase 3: Association & Synthesis ---
def calculate_distance(bbox1, bbox2):
    """Calculates the Euclidean distance between the centers of two bounding boxes."""
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def associate_images_with_products(products_data, images_list, text_blocks):
    """
    Associates images to products based on the geometric proximity of their bounding boxes.
    Uses product_name, sku_description, and product_code for better matching with the new format.
    """
    if not images_list:
        return products_data

    for product in products_data["products"]:
        product_name = product.get("product_name", "")
        if not product_name:
            continue

        # Collect all possible text identifiers from the product and its SKUs
        search_terms = [product_name]
        
        # Add SKU information for better matching
        skus = product.get("skus", [])
        for sku in skus:
            sku_description = sku.get("sku_description", "")
            product_code = sku.get("product_code", "")
            
            if sku_description and sku_description not in search_terms:
                search_terms.append(sku_description)
            if product_code and product_code != "N/A" and product_code not in search_terms:
                search_terms.append(product_code)

        # Find the bounding box for the product using multiple matching strategies
        product_bbox = None
        for block in text_blocks:
            block_text_lower = block["text"].lower()
            
            # Try matching with product name first (most reliable)
            if product_name.lower() in block_text_lower:
                product_bbox = block["bbox"]
                break
            
            # Try matching with SKU description
            for term in search_terms[1:]:  # Skip product_name as we already checked it
                if term.lower() in block_text_lower:
                    product_bbox = block["bbox"]
                    break
            
            if product_bbox:
                break
        
        # If no exact match found, try partial matching with key words from product name
        if not product_bbox and product_name:
            # Extract key words (remove common words, keep product identifiers)
            words = [w for w in product_name.split() if len(w) > 2 and w.lower() not in ['the', 'and', 'for', 'with']]
            for block in text_blocks:
                block_text_lower = block["text"].lower()
                # Check if multiple key words appear in the block
                matches = sum(1 for word in words if word.lower() in block_text_lower)
                if matches >= 2:  # At least 2 key words match
                    product_bbox = block["bbox"]
                    break
        
        if not product_bbox:
            continue

        # Find the closest image to this product's text
        min_distance = float('inf')
        closest_image_path = None
        for image in images_list:
            dist = calculate_distance(product_bbox, image["bbox"])
            if dist < min_distance:
                min_distance = dist
                closest_image_path = image["path"]
        
        product["associated_image"] = closest_image_path

    return products_data


# --- Main Pipeline Orchestrator ---
def process_catalogue(pdf_path):
    """
    Main function to run the entire extraction and association pipeline.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return

    print(f"Starting processing for '{pdf_path}'...")
    doc = fitz.open(pdf_path)
    
    all_products_data = []

    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}/{len(doc)}...")

        # Phase 1: Decompose PDF page
        images, text_blocks = extract_page_elements(page, page_num)
        
        # Combine text for Gemini
        full_page_text = "\n".join([block['text'] for block in text_blocks if block['text']])
        
        if not full_page_text.strip():
            print("  - No text content found on this page. Skipping.")
            continue

        # Phase 2: Get structured data from Gemini
        print("  - Sending text to Gemini for structuring...")
        structured_data = get_structured_data_from_gemini(full_page_text)
        
        if not structured_data or not structured_data.get("products"):
            print("  - Gemini found no products on this page.")
            continue
        
        print(f"  - Gemini extracted {len(structured_data['products'])} product(s).")

        # Phase 3: Associate images with the structured data
        print("  - Associating images with products...")
        final_data_for_page = associate_images_with_products(structured_data, images, text_blocks)
        
        # Add source page number for reference
        for product in final_data_for_page.get("products", []):
            product["source_page"] = page_num + 1
        
        all_products_data.extend(final_data_for_page.get("products", []))

    # Save the final consolidated JSON
    output_json_path = os.path.join(JSON_DIR, "catalogue_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_products_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("Processing complete!")
    print(f"Total products extracted: {len(all_products_data)}")
    print(f"Images saved to: '{IMAGE_DIR}'")
    print(f"Structured data saved to: '{output_json_path}'")
    print("="*50)


if __name__ == "__main__":
    # Replace with the path to your PDF file
    PDF_FILE_PATH = "/Users/ronballer/Downloads/Kich Price List 2025 - Ver 02_removed.pdf"
    process_catalogue(PDF_FILE_PATH)