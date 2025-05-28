import os
from dotenv import load_dotenv
from groq import Groq
import json
from supabase import create_client, Client
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import re
from uuid import UUID
import requests
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import io as skio
import logging
import base64
import aiohttp
import argparse

load_dotenv()

# Initialize clients
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Warning: GROQ_API_KEY not set in .env. Fraud detection will be limited.")
    groq_client = None
else:
    groq_client = Groq(api_key=groq_api_key)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in .env. Database storage will not work.")
    supabase: Optional[Client] = None
else:
    supabase = create_client(supabase_url, supabase_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptFraudDetector:
    def __init__(self):
        self.risk_factors = []
        self.verification_results = {}
        self.image_analysis_results = {}
        self.online_verification_results = {}
        self.original_text = ""
        self.enhanced_text = ""
        self.llm_ocr_text = ""
        self.llm_structured_data = {}
        self.expense_category = None

    async def analyze_receipt(self, expense_id: UUID, file_url: str) -> Dict[str, Any]:
        """
        Analyze a receipt for potential fraud using multiple methods.
        
        Args:
            expense_id: UUID of the expense record
            file_url: URL of the receipt image/document
            
        Returns:
            Dictionary containing fraud analysis results
        """
        try:
            # Get expense data from database
            expense_data = self._get_expense_data(expense_id)
            if not expense_data:
                raise ValueError(f"Expense {expense_id} not found")

            # Determine expense category
            self.expense_category = expense_data.get('category', '').lower()

            # Run all checks in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._llm_analysis, expense_data, file_url),
                    executor.submit(self._image_analysis, file_url),
                    executor.submit(self._online_verification, expense_data),
                    executor.submit(self._pattern_analysis, expense_data)
                ]
                
                # Run category specific verification separately since it's async
                category_verification = await self._category_specific_verification(expense_data)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if isinstance(result, dict):
                            if 'risk_factors' in result:
                                self.risk_factors.extend(result['risk_factors'])
                            if 'verification_results' in result:
                                self.verification_results.update(result['verification_results'])
                            if 'image_analysis_results' in result:
                                self.image_analysis_results.update(result['image_analysis_results'])
                            if 'online_verification_results' in result:
                                self.online_verification_results.update(result['online_verification_results'])
                    except Exception as e:
                        print(f"Error in parallel check: {e}")

                # Add category verification results
                if isinstance(category_verification, dict):
                    if 'risk_factors' in category_verification:
                        self.risk_factors.extend(category_verification['risk_factors'])
                    if 'verification_results' in category_verification:
                        self.verification_results.update(category_verification['verification_results'])

            # Calculate overall risk score
            overall_risk_score = self._calculate_risk_score()
            fraud_probability = self._calculate_fraud_probability()

            # Generate summary
            summary = self._generate_summary(overall_risk_score, fraud_probability)

            # Store results in database
            result_id = self._store_fraud_check_results(
                expense_id,
                overall_risk_score,
                fraud_probability
            )

            return {
                "fraud_check_id": result_id,
                "overall_risk_score": overall_risk_score,
                "fraud_probability": fraud_probability,
                "risk_factors": self.risk_factors,
                "verification_results": self.verification_results,
                "image_analysis_results": self.image_analysis_results,
                "online_verification_results": self.online_verification_results,
                "summary": summary
            }

        except Exception as e:
            print(f"Error in fraud analysis: {e}")
            return {}

    def _get_expense_data(self, expense_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve expense data from database"""
        if supabase is None:
            return None
            
        try:
            result = supabase.table('expenses').select('*').eq('id', str(expense_id)).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error retrieving expense data: {e}")
            return None

    def _llm_analysis(self, expense_data: Dict[str, Any], file_url: str) -> Dict[str, Any]:
        """Use LLM to analyze receipt content for suspicious patterns"""
        if groq_client is None:
            return {}

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI specialized in detecting fraudulent receipts.
                        Analyze the provided receipt and look for:
                        1. Inconsistent dates, amounts, or vendor information
                        2. Unusual patterns in the receipt format
                        3. Suspicious modifications or alterations
                        4. Mismatches between receipt details and expense data
                        5. Common fraud indicators
                        
                        Format your response as a JSON object with:
                        - risk_factors: List of identified risk factors
                        - verification_results: Detailed analysis of each aspect
                        - confidence_score: Your confidence in the analysis (0-1)"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please analyze this receipt for potential fraud. Expense data: {json.dumps(expense_data)}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": file_url}
                            }
                        ]
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={"type": "json_object"}
            )
            
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {}

    def _image_analysis(self, file_url: str) -> Dict[str, Any]:
        """Analyze receipt image for signs of tampering or forgery using open-source tools"""
        results = {}
        
        try:
            # Download image
            response = requests.get(file_url)
            image_data = response.content
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image analysis
            results['image_quality'] = self._analyze_image_quality(image)
            
            # OCR analysis with multiple preprocessing steps
            results['ocr_analysis'] = self._enhanced_ocr_analysis(image)
            
            # Check for image manipulation
            results['manipulation_indicators'] = self._check_image_manipulation(image)
            
            # Extract and verify text patterns
            results['text_pattern_analysis'] = self._analyze_text_patterns(results['ocr_analysis'])
            
            return {"image_analysis_results": results}
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {}

    def _online_verification(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify receipt details against online sources"""
        results = {
            'vendor_verification': self._verify_vendor(expense_data.get('vendor_name')),
            'amount_verification': self._verify_amount(
                expense_data.get('amount'),
                expense_data.get('vendor_name'),
                expense_data.get('transaction_date')
            ),
            'date_verification': self._verify_date(
                expense_data.get('transaction_date'),
                expense_data.get('vendor_name')
            )
        }
        return {"online_verification_results": results}

    def _pattern_analysis(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze expense patterns for suspicious activity"""
        try:
            # Get user's expense history
            user_id = expense_data.get('user_id')
            if not user_id or not supabase:
                return {}

            # Get recent expenses
            recent_expenses = supabase.table('expenses')\
                .select('*')\
                .eq('user_id', str(user_id))\
                .order('transaction_date', desc=True)\
                .limit(10)\
                .execute()

            if not recent_expenses.data:
                return {}

            # Analyze patterns
            patterns = {
                'unusual_amounts': self._check_unusual_amounts(expense_data, recent_expenses.data),
                'frequency_patterns': self._check_frequency_patterns(expense_data, recent_expenses.data),
                'vendor_patterns': self._check_vendor_patterns(expense_data, recent_expenses.data)
            }

            return {
                "risk_factors": self._convert_patterns_to_risk_factors(patterns),
                "verification_results": {"pattern_analysis": patterns}
            }
        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            return {}

    async def _category_specific_verification(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform category-specific verification using LLM and online data"""
        if groq_client is None:
            return {}

        try:
            category = self.expense_category
            if not category:
                return {}

            # Get verification tools
            tools = self._get_verification_tools()
            
            # Prepare category-specific prompt
            prompt = self._get_category_specific_prompt(category, expense_data)
            if not prompt:
                return {}

            # Make online verification call
            verification_results = await self._make_online_verification_call(prompt, tools)
            
            # Add category-specific risk factors
            if verification_results.get('verification_results'):
                self.verification_results.update({
                    f"{category}_verification": verification_results['verification_results']
                })
            
            # Add risk factors based on verification results
            risk_factors = self._extract_risk_factors_from_verification(
                verification_results['verification_results'],
                category
            )
            self.risk_factors.extend(risk_factors)

            return verification_results

        except Exception as e:
            print(f"Error in category-specific verification: {e}")
            return {}

    def _get_category_specific_prompt(self, category: str, expense_data: Dict[str, Any]) -> Optional[str]:
        """Get category-specific prompt for LLM verification"""
        prompts = {
            'travel': """You are an AI specialized in verifying travel expenses.
            Analyze the provided travel expense and verify against online data:
            1. Distance-based pricing:
               - Calculate expected fare based on route distance
               - Compare with actual fare
               - Check for common routes and typical pricing
            2. Time-based factors:
               - Verify if the time of travel affects pricing
               - Check for peak/off-peak rates
               - Validate travel duration
            3. Service provider verification:
               - Verify if the service provider operates in the claimed area
               - Check typical pricing for this provider
               - Validate service type (e.g., economy, premium)
            
            Format your response as a JSON object with:
            - risk_factors: List of identified risk factors
            - verification_results: Detailed analysis including:
              * expected_price_range: Min and max expected price
              * price_discrepancy: Difference between expected and actual
              * route_verification: Whether the route is typical
              * time_verification: Whether the time/date is reasonable
              * provider_verification: Whether the provider is legitimate
            - confidence_score: Your confidence in the verification (0-1)""",

            'hotel': """You are an AI specialized in verifying hotel expenses.
            Analyze the provided hotel expense and verify against online data:
            1. Location-based pricing:
               - Check typical rates for the area
               - Verify if the location matches the claimed area
               - Compare with similar hotels in the area
            2. Date-based factors:
               - Check if the dates affect pricing (season, events)
               - Verify if the hotel was open on those dates
               - Validate length of stay
            3. Room and service verification:
               - Verify room type pricing
               - Check included services and amenities
               - Validate additional charges
            
            Format your response as a JSON object with:
            - risk_factors: List of identified risk factors
            - verification_results: Detailed analysis including:
              * expected_price_range: Min and max expected price
              * price_discrepancy: Difference between expected and actual
              * location_verification: Whether the location is legitimate
              * date_verification: Whether the dates are reasonable
              * service_verification: Whether the services are typical
            - confidence_score: Your confidence in the verification (0-1)""",

            'food': """You are an AI specialized in verifying food and dining expenses.
            Analyze the provided dining expense and verify against online data:
            1. Restaurant verification:
               - Check if the restaurant exists in the claimed location
               - Verify typical pricing for this establishment
               - Compare with similar restaurants in the area
            2. Menu-based verification:
               - Check if the items ordered are on the menu
               - Verify typical prices for ordered items
               - Validate portion sizes and quantities
            3. Time and date verification:
               - Check if the restaurant was open at that time
               - Verify if the date affects pricing (special menus, events)
               - Validate if the amount is typical for that time of day
            
            Format your response as a JSON object with:
            - risk_factors: List of identified risk factors
            - verification_results: Detailed analysis including:
              * expected_price_range: Min and max expected price
              * price_discrepancy: Difference between expected and actual
              * restaurant_verification: Whether the restaurant is legitimate
              * menu_verification: Whether the items and prices are typical
              * time_verification: Whether the time/date is reasonable
            - confidence_score: Your confidence in the verification (0-1)"""
        }

        return prompts.get(category)

    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score based on all analyses"""
        weights = {
            'llm_analysis': 0.3,  # Reduced from 0.4
            'image_analysis': 0.25,  # Reduced from 0.3
            'online_verification': 0.15,  # Reduced from 0.2
            'pattern_analysis': 0.1,
            'category_specific': 0.2  # New weight for category-specific checks
        }
        
        scores = {
            'llm_analysis': self._get_llm_risk_score(),
            'image_analysis': self._get_image_risk_score(),
            'online_verification': self._get_verification_risk_score(),
            'pattern_analysis': self._get_pattern_risk_score(),
            'category_specific': self._get_category_specific_risk_score()  # New score
        }
        
        return sum(score * weights[category] for category, score in scores.items())

    def _calculate_fraud_probability(self) -> float:
        """Calculate probability of fraud based on risk factors and verification results"""
        # Implementation depends on your specific requirements
        # This is a simplified version
        risk_score = self._calculate_risk_score()
        verification_confidence = self._get_verification_confidence()
        
        return (risk_score * 0.7) + (verification_confidence * 0.3)

    def _store_fraud_check_results(
        self,
        expense_id: UUID,
        overall_risk_score: float,
        fraud_probability: float
    ) -> Optional[str]:
        """Store fraud check results in database"""
        if supabase is None:
            return None
            
        try:
            result = supabase.table('receipt_fraud_checks').insert({
                'expense_id': str(expense_id),
                'overall_risk_score': overall_risk_score,
                'fraud_probability': fraud_probability,
                'risk_factors': self.risk_factors,
                'verification_results': self.verification_results,
                'image_analysis_results': self.image_analysis_results,
                'online_verification_results': self.online_verification_results
            }).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
        except Exception as e:
            print(f"Error storing fraud check results: {e}")
            return None

    # Helper methods for various analyses
    def _analyze_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image quality metrics using open-source tools"""
        try:
            # Convert PIL Image to numpy array for OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Calculate image quality metrics
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Check for noise
            noise = self._estimate_noise(gray)
            
            # Check for compression artifacts
            compression_score = self._detect_compression_artifacts(gray)
            
            return {
                'blur_score': float(blur_score),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'noise_level': float(noise),
                'compression_artifacts': float(compression_score),
                'quality_assessment': self._assess_image_quality(
                    blur_score, brightness, contrast, noise, compression_score
                )
            }
        except Exception as e:
            logger.error(f"Error in image quality analysis: {e}")
            return {}

    def _enhanced_ocr_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Perform enhanced OCR analysis using multiple methods (Tesseract and LLM)"""
        results = {}
        
        try:
            # Store original image
            original_image = image.copy()
            
            # 1. Tesseract OCR with multiple preprocessing techniques
            tesseract_results = self._tesseract_ocr_analysis(image)
            
            # 2. LLM-based OCR analysis
            llm_results = self._llm_ocr_analysis(image)
            
            # Compare and combine results
            results['tesseract_analysis'] = tesseract_results
            results['llm_analysis'] = llm_results
            results['combined_analysis'] = self._combine_ocr_results(tesseract_results, llm_results)
            
            # Store all versions of extracted text
            self.original_text = tesseract_results.get('original_text', '')
            self.enhanced_text = tesseract_results.get('best_result', {}).get('text', '')
            self.llm_ocr_text = llm_results.get('extracted_text', '')
            self.llm_structured_data = llm_results.get('structured_data', {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced OCR analysis: {e}")
            return {}

    def _tesseract_ocr_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Perform OCR analysis using Tesseract with multiple preprocessing techniques"""
        results = {}
        
        try:
            # Store original text
            original_text = pytesseract.image_to_string(image)
            results['original_text'] = original_text
            
            # Try different preprocessing techniques
            preprocessing_results = []
            
            # 1. Basic preprocessing
            basic_processed = self._preprocess_image(image)
            basic_text = pytesseract.image_to_string(basic_processed)
            preprocessing_results.append({
                'method': 'basic',
                'text': basic_text,
                'confidence': self._calculate_ocr_confidence(basic_text)
            })
            
            # 2. Enhanced contrast
            contrast_enhanced = self._enhance_contrast(image)
            contrast_text = pytesseract.image_to_string(contrast_enhanced)
            preprocessing_results.append({
                'method': 'contrast_enhanced',
                'text': contrast_text,
                'confidence': self._calculate_ocr_confidence(contrast_text)
            })
            
            # 3. Denoised
            denoised = self._denoise_image(image)
            denoised_text = pytesseract.image_to_string(denoised)
            preprocessing_results.append({
                'method': 'denoised',
                'text': denoised_text,
                'confidence': self._calculate_ocr_confidence(denoised_text)
            })
            
            # Select best result
            best_result = max(preprocessing_results, key=lambda x: x['confidence'])
            results['best_result'] = best_result
            results['all_preprocessing_results'] = preprocessing_results
            
            # Extract structured data
            results['structured_data'] = self._extract_structured_data(best_result['text'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Tesseract OCR analysis: {e}")
            return {}

    def _llm_ocr_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Perform OCR analysis using LLM (Groq)"""
        if groq_client is None:
            return {}
            
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare prompt for LLM
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI specialized in extracting information from receipts.
                        Analyze the provided receipt image and extract:
                        1. All text content
                        2. Structured data including:
                           - Date
                           - Total amount
                           - Tax amount
                           - Vendor name
                           - Items with descriptions and amounts
                           - Payment method
                           - Document ID/Reference number
                        
                        Format your response as a JSON object with:
                        - extracted_text: The complete text content
                        - structured_data: Object containing all extracted fields
                        - confidence_score: Your confidence in the extraction (0-1)
                        - extraction_notes: Any notes about the extraction process"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract all information from this receipt image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            llm_response = json.loads(chat_completion.choices[0].message.content)
            
            # Add OCR method identifier
            llm_response['ocr_method'] = 'llm'
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error in LLM OCR analysis: {e}")
            return {}

    def _combine_ocr_results(self, tesseract_results: Dict[str, Any], llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine and validate results from both OCR methods"""
        combined = {
            'extracted_text': '',
            'structured_data': {},
            'confidence_score': 0.0,
            'validation_results': {},
            'discrepancies': []
        }
        
        try:
            # Get best text from each method
            tesseract_text = tesseract_results.get('best_result', {}).get('text', '')
            llm_text = llm_results.get('extracted_text', '')
            
            # Get structured data from each method
            tesseract_data = tesseract_results.get('structured_data', {})
            llm_data = llm_results.get('structured_data', {})
            
            # Compare and validate results
            validation = self._validate_ocr_results(tesseract_data, llm_data)
            
            # Select best text based on confidence
            tesseract_confidence = tesseract_results.get('best_result', {}).get('confidence', 0.0)
            llm_confidence = llm_results.get('confidence_score', 0.0)
            
            if llm_confidence > tesseract_confidence:
                combined['extracted_text'] = llm_text
                combined['structured_data'] = llm_data
                combined['confidence_score'] = llm_confidence
            else:
                combined['extracted_text'] = tesseract_text
                combined['structured_data'] = tesseract_data
                combined['confidence_score'] = tesseract_confidence
            
            # Add validation results
            combined['validation_results'] = validation
            
            # Check for discrepancies
            discrepancies = self._find_discrepancies(tesseract_data, llm_data)
            if discrepancies:
                combined['discrepancies'] = discrepancies
                # Adjust confidence based on discrepancies
                combined['confidence_score'] *= (1 - len(discrepancies) * 0.1)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining OCR results: {e}")
            return combined

    def _validate_ocr_results(self, tesseract_data: Dict[str, Any], llm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and compare structured data from both OCR methods"""
        validation = {
            'date_match': False,
            'amount_match': False,
            'vendor_match': False,
            'tax_match': False,
            'overall_match_score': 0.0
        }
        
        try:
            # Compare dates
            if tesseract_data.get('date') and llm_data.get('date'):
                validation['date_match'] = self._normalize_date(tesseract_data['date']) == self._normalize_date(llm_data['date'])
            
            # Compare amounts
            if tesseract_data.get('total_amount') and llm_data.get('total_amount'):
                validation['amount_match'] = abs(float(tesseract_data['total_amount']) - float(llm_data['total_amount'])) < 0.01
            
            # Compare vendor names
            if tesseract_data.get('vendor_name') and llm_data.get('vendor_name'):
                validation['vendor_match'] = self._normalize_text(tesseract_data['vendor_name']) == self._normalize_text(llm_data['vendor_name'])
            
            # Compare tax amounts
            if tesseract_data.get('tax_amount') and llm_data.get('tax_amount'):
                validation['tax_match'] = abs(float(tesseract_data['tax_amount']) - float(llm_data['tax_amount'])) < 0.01
            
            # Calculate overall match score
            matches = sum(validation.values())
            validation['overall_match_score'] = matches / len(validation)
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating OCR results: {e}")
            return validation

    def _find_discrepancies(self, tesseract_data: Dict[str, Any], llm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find discrepancies between OCR results"""
        discrepancies = []
        
        try:
            # Check date discrepancy
            if tesseract_data.get('date') and llm_data.get('date'):
                if self._normalize_date(tesseract_data['date']) != self._normalize_date(llm_data['date']):
                    discrepancies.append({
                        'field': 'date',
                        'tesseract_value': tesseract_data['date'],
                        'llm_value': llm_data['date']
                    })
            
            # Check amount discrepancy
            if tesseract_data.get('total_amount') and llm_data.get('total_amount'):
                if abs(float(tesseract_data['total_amount']) - float(llm_data['total_amount'])) >= 0.01:
                    discrepancies.append({
                        'field': 'total_amount',
                        'tesseract_value': tesseract_data['total_amount'],
                        'llm_value': llm_data['total_amount']
                    })
            
            # Check vendor name discrepancy
            if tesseract_data.get('vendor_name') and llm_data.get('vendor_name'):
                if self._normalize_text(tesseract_data['vendor_name']) != self._normalize_text(llm_data['vendor_name']):
                    discrepancies.append({
                        'field': 'vendor_name',
                        'tesseract_value': tesseract_data['vendor_name'],
                        'llm_value': llm_data['vendor_name']
                    })
            
            # Check tax amount discrepancy
            if tesseract_data.get('tax_amount') and llm_data.get('tax_amount'):
                if abs(float(tesseract_data['tax_amount']) - float(llm_data['tax_amount'])) >= 0.01:
                    discrepancies.append({
                        'field': 'tax_amount',
                        'tesseract_value': tesseract_data['tax_amount'],
                        'llm_value': llm_data['tax_amount']
                    })
            
            return discrepancies
            
        except Exception as e:
            logger.error(f"Error finding discrepancies: {e}")
            return discrepancies

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format"""
        try:
            # Try different date formats
            formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y']
            for fmt in formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str
        except Exception:
            return date_str

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        try:
            # Convert to lowercase and remove special characters
            normalized = re.sub(r'[^a-z0-9]', '', text.lower())
            return normalized
        except Exception:
            return text

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Basic image preprocessing for OCR"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Normalize contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        return image

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast for better OCR"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply adaptive histogram equalization
        img_array = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_array)
        
        return Image.fromarray(enhanced)

    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(img_array)
        
        return Image.fromarray(denoised)

    def _calculate_ocr_confidence(self, text: str) -> float:
        """Calculate confidence score for OCR results"""
        if not text.strip():
            return 0.0
            
        # Check for common receipt elements
        receipt_elements = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # Date
            r'\$\d+\.\d{2}',  # Dollar amount
            r'TOTAL|SUBTOTAL|TAX',  # Common receipt words
            r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}'  # Credit card number
        ]
        
        matches = sum(1 for pattern in receipt_elements if re.search(pattern, text))
        return min(1.0, matches / len(receipt_elements))

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from OCR text"""
        data = {
            'date': None,
            'total_amount': None,
            'tax_amount': None,
            'vendor_name': None,
            'items': []
        }
        
        try:
            # Extract date
            date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)
            if date_match:
                data['date'] = date_match.group()
            
            # Extract total amount
            total_match = re.search(r'TOTAL\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
            if total_match:
                data['total_amount'] = float(total_match.group(1))
            
            # Extract tax amount
            tax_match = re.search(r'TAX\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
            if tax_match:
                data['tax_amount'] = float(tax_match.group(1))
            
            # Extract vendor name (usually at the top of receipt)
            lines = text.split('\n')
            if lines:
                data['vendor_name'] = lines[0].strip()
            
            # Extract items (simplified version)
            item_pattern = r'([A-Za-z\s]+)\s+\$?(\d+\.\d{2})'
            for line in lines:
                item_match = re.search(item_pattern, line)
                if item_match:
                    data['items'].append({
                        'description': item_match.group(1).strip(),
                        'amount': float(item_match.group(2))
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return data

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate image noise level"""
        try:
            # Apply median filter
            median = cv2.medianBlur(image, 3)
            # Calculate difference
            diff = cv2.absdiff(image, median)
            # Return noise estimate
            return float(np.mean(diff))
        except Exception as e:
            logger.error(f"Error estimating noise: {e}")
            return 0.0

    def _detect_compression_artifacts(self, image: np.ndarray) -> float:
        """Detect JPEG compression artifacts"""
        try:
            # Apply DCT transform
            dct = cv2.dct(np.float32(image))
            # Calculate high-frequency components
            hf_energy = np.sum(np.abs(dct[8:, 8:]))
            # Normalize
            return float(hf_energy / (image.shape[0] * image.shape[1]))
        except Exception as e:
            logger.error(f"Error detecting compression artifacts: {e}")
            return 0.0

    def _assess_image_quality(self, blur: float, brightness: float, 
                            contrast: float, noise: float, compression: float) -> str:
        """Assess overall image quality"""
        if blur < 100:
            return "Poor quality - Image is too blurry"
        elif brightness < 50 or brightness > 200:
            return "Poor quality - Incorrect brightness"
        elif contrast < 30:
            return "Poor quality - Low contrast"
        elif noise > 20:
            return "Poor quality - High noise level"
        elif compression > 0.5:
            return "Poor quality - Heavy compression artifacts"
        else:
            return "Good quality"

    def _check_image_manipulation(self, image: Image.Image) -> List[str]:
        """Check for signs of image manipulation using various techniques"""
        manipulation_indicators = []
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # 1. Check for copy-move forgery
            if self._detect_copy_move(img_array):
                manipulation_indicators.append("Possible copy-move forgery detected")
            
            # 2. Check for inconsistent JPEG compression
            if self._check_jpeg_consistency(img_array):
                manipulation_indicators.append("Inconsistent JPEG compression detected")
            
            # 3. Check for noise inconsistencies
            if self._check_noise_inconsistency(img_array):
                manipulation_indicators.append("Inconsistent noise patterns detected")
            
            # 4. Check for edge inconsistencies
            if self._check_edge_inconsistency(img_array):
                manipulation_indicators.append("Inconsistent edge patterns detected")
            
            # 5. Check for metadata anomalies
            metadata_issues = self._check_metadata(image)
            manipulation_indicators.extend(metadata_issues)
            
            return manipulation_indicators
            
        except Exception as e:
            logger.error(f"Error in image manipulation detection: {e}")
            return []

    def _detect_copy_move(self, image: np.ndarray) -> bool:
        """Detect copy-move forgery using block matching"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Divide image into blocks
            block_size = 16
            height, width = gray.shape
            blocks = []
            
            for i in range(0, height - block_size, block_size):
                for j in range(0, width - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    blocks.append((block, (i, j)))
            
            # Compare blocks
            for i, (block1, pos1) in enumerate(blocks):
                for block2, pos2 in blocks[i+1:]:
                    # Calculate similarity
                    similarity = ssim(block1, block2)
                    if similarity > 0.95:  # High similarity threshold
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in copy-move detection: {e}")
            return False

    def _check_jpeg_consistency(self, image: np.ndarray) -> bool:
        """Check for inconsistent JPEG compression"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply DCT transform
            dct = cv2.dct(np.float32(gray))
            
            # Analyze DCT coefficients
            dct_blocks = np.array_split(dct, 8, axis=0)
            dct_blocks = [np.array_split(block, 8, axis=1) for block in dct_blocks]
            
            # Check for inconsistencies in DCT coefficients
            block_energies = []
            for row in dct_blocks:
                for block in row:
                    energy = np.sum(np.abs(block))
                    block_energies.append(energy)
            
            # Calculate variance of block energies
            energy_variance = np.var(block_energies)
            
            return energy_variance > 1000  # Threshold for inconsistency
            
        except Exception as e:
            logger.error(f"Error in JPEG consistency check: {e}")
            return False

    def _check_noise_inconsistency(self, image: np.ndarray) -> bool:
        """Check for inconsistent noise patterns"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Divide image into regions
            regions = np.array_split(gray, 4, axis=0)
            regions = [np.array_split(region, 4, axis=1) for region in regions]
            
            # Calculate noise level for each region
            noise_levels = []
            for row in regions:
                for region in row:
                    noise = self._estimate_noise(region)
                    noise_levels.append(noise)
            
            # Check for significant variations in noise levels
            noise_variance = np.var(noise_levels)
            return noise_variance > 50  # Threshold for inconsistency
            
        except Exception as e:
            logger.error(f"Error in noise inconsistency check: {e}")
            return False

    def _check_edge_inconsistency(self, image: np.ndarray) -> bool:
        """Check for inconsistent edge patterns"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect edges
            edges = cv2.Canny(gray, 100, 200)
            
            # Divide edge image into regions
            regions = np.array_split(edges, 4, axis=0)
            regions = [np.array_split(region, 4, axis=1) for region in regions]
            
            # Calculate edge density for each region
            edge_densities = []
            for row in regions:
                for region in row:
                    density = np.sum(region) / (region.shape[0] * region.shape[1])
                    edge_densities.append(density)
            
            # Check for significant variations in edge densities
            density_variance = np.var(edge_densities)
            return density_variance > 0.1  # Threshold for inconsistency
            
        except Exception as e:
            logger.error(f"Error in edge inconsistency check: {e}")
            return False

    def _check_metadata(self, image: Image.Image) -> List[str]:
        """Check image metadata for anomalies"""
        anomalies = []
        try:
            # Check for basic metadata
            if not image.info:
                anomalies.append("No metadata found")
            
            # Check for suspicious software
            if 'Software' in image.info:
                software = image.info['Software'].lower()
                suspicious_software = ['photoshop', 'gimp', 'paint.net']
                if any(s in software for s in suspicious_software):
                    anomalies.append(f"Suspicious editing software detected: {software}")
            
            # Check for multiple saves
            if 'Comment' in image.info and 'saved' in image.info['Comment'].lower():
                anomalies.append("Image appears to have been saved multiple times")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error checking metadata: {e}")
            return ["Error checking metadata"]

    def _analyze_text_patterns(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text patterns for potential fraud indicators"""
        results = {
            'suspicious_patterns': [],
            'text_consistency': {},
            'format_analysis': {}
        }
        
        try:
            text = ocr_results.get('extracted_text', '')
            if not text:
                return results
            
            # Check for suspicious patterns
            patterns = {
                'multiple_decimals': r'\d+\.\d+\.\d+',  # Multiple decimal points
                'inconsistent_spacing': r'\d{2,}\s{2,}\d{2,}',  # Inconsistent number spacing
                'suspicious_chars': r'[^a-zA-Z0-9\s\.\,\$\-\+\/]',  # Unusual characters
                'repeated_numbers': r'(\d+)\1{2,}'  # Repeated number sequences
            }
            
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    results['suspicious_patterns'].append({
                        'pattern': pattern_name,
                        'matches': matches
                    })
            
            # Analyze text consistency
            lines = text.split('\n')
            results['text_consistency'] = {
                'line_count': len(lines),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'empty_lines': sum(1 for line in lines if not line.strip()),
                'inconsistent_indentation': self._check_indentation_consistency(lines)
            }
            
            # Analyze receipt format
            results['format_analysis'] = {
                'has_header': bool(re.search(r'RECEIPT|INVOICE|BILL', text, re.IGNORECASE)),
                'has_footer': bool(re.search(r'THANK YOU|TOTAL|SUBTOTAL', text, re.IGNORECASE)),
                'has_date': bool(re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)),
                'has_amount': bool(re.search(r'\$\d+\.\d{2}', text))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing text patterns: {e}")
            return results

    def _check_indentation_consistency(self, lines: List[str]) -> bool:
        """Check for inconsistent indentation in text"""
        try:
            # Get indentation for each line
            indentations = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            
            if not indentations:
                return False
            
            # Check if indentation is consistent
            return np.std(indentations) > 2  # Threshold for inconsistency
            
        except Exception as e:
            logger.error(f"Error checking indentation consistency: {e}")
            return False

    def _verify_vendor(self, vendor_name: str) -> Dict[str, Any]:
        """Verify vendor information online"""
        # Implementation for vendor verification
        return {}

    def _verify_amount(self, amount: float, vendor: str, date: str) -> Dict[str, Any]:
        """Verify amount against vendor's pricing"""
        # Implementation for amount verification
        return {}

    def _verify_date(self, date: str, vendor: str) -> Dict[str, Any]:
        """Verify date against vendor's operating hours"""
        # Implementation for date verification
        return {}

    def _check_unusual_amounts(self, current_expense: Dict[str, Any], recent_expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for unusual amount patterns"""
        # Implementation for amount pattern analysis
        return {}

    def _check_frequency_patterns(self, current_expense: Dict[str, Any], recent_expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for unusual frequency patterns"""
        # Implementation for frequency pattern analysis
        return {}

    def _check_vendor_patterns(self, current_expense: Dict[str, Any], recent_expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for unusual vendor patterns"""
        # Implementation for vendor pattern analysis
        return {}

    def _convert_patterns_to_risk_factors(self, patterns: Dict[str, Any]) -> List[str]:
        """Convert pattern analysis results to risk factors"""
        # Implementation for converting patterns to risk factors
        return []

    def _get_llm_risk_score(self) -> float:
        """Extract risk score from LLM analysis"""
        # Implementation for LLM risk score calculation
        return 0.0

    def _get_image_risk_score(self) -> float:
        """Calculate risk score from image analysis"""
        # Implementation for image risk score calculation
        return 0.0

    def _get_verification_risk_score(self) -> float:
        """Calculate risk score from online verification"""
        # Implementation for verification risk score calculation
        return 0.0

    def _get_pattern_risk_score(self) -> float:
        """Calculate risk score from pattern analysis"""
        # Implementation for pattern risk score calculation
        return 0.0

    def _get_verification_confidence(self) -> float:
        """Calculate confidence in verification results"""
        # Implementation for verification confidence calculation
        return 0.0

    def _get_category_specific_risk_score(self) -> float:
        """Calculate risk score from category-specific verification"""
        try:
            category = self.expense_category
            if not category:
                return 0.0

            verification_results = self.verification_results.get(f"{category}_verification", {})
            if not verification_results:
                return 0.0

            # Calculate risk based on verification results
            risk_factors = []
            
            # Check price discrepancy
            if 'price_discrepancy' in verification_results:
                discrepancy = abs(verification_results['price_discrepancy'])
                if discrepancy > 0.5:  # More than 50% difference
                    risk_factors.append(1.0)
                elif discrepancy > 0.2:  # More than 20% difference
                    risk_factors.append(0.7)
                elif discrepancy > 0.1:  # More than 10% difference
                    risk_factors.append(0.4)

            # Check location/route verification
            if 'location_verification' in verification_results and not verification_results['location_verification']:
                risk_factors.append(0.8)
            elif 'route_verification' in verification_results and not verification_results['route_verification']:
                risk_factors.append(0.8)

            # Check time/date verification
            if 'time_verification' in verification_results and not verification_results['time_verification']:
                risk_factors.append(0.6)
            elif 'date_verification' in verification_results and not verification_results['date_verification']:
                risk_factors.append(0.6)

            # Check service/restaurant verification
            if 'service_verification' in verification_results and not verification_results['service_verification']:
                risk_factors.append(0.7)
            elif 'restaurant_verification' in verification_results and not verification_results['restaurant_verification']:
                risk_factors.append(0.7)

            # Check menu verification for food expenses
            if category == 'food' and 'menu_verification' in verification_results and not verification_results['menu_verification']:
                risk_factors.append(0.5)

            # Return average of risk factors, or 0.0 if no factors found
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0

        except Exception as e:
            print(f"Error calculating category-specific risk score: {e}")
            return 0.0

    def _make_online_verification_call(self, prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make an API call to Groq with tool calling capabilities for online verification"""
        if groq_client is None:
            return {}

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI specialized in verifying expense data using online tools.
                        You have access to various tools to verify:
                        1. Vendor information and pricing
                        2. Location-based rates and services
                        3. Historical pricing data
                        4. Operating hours and availability
                        
                        Use the provided tools to verify the expense data and return detailed results."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                tools=tools,
                tool_choice="auto"
            )

            # Process the response and tool calls
            response = chat_completion.choices[0].message
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

            results = {
                'verification_results': {},
                'tool_calls': [],
                'confidence_score': 0.0
            }

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute the tool call based on the function name
                if tool_name == 'search_vendor_info':
                    vendor_info = self._search_vendor_info(tool_args.get('vendor_name'))
                    results['verification_results']['vendor_info'] = vendor_info
                elif tool_name == 'check_pricing':
                    pricing_info = self._check_pricing(
                        tool_args.get('vendor_name'),
                        tool_args.get('service_type'),
                        tool_args.get('location'),
                        tool_args.get('date')
                    )
                    results['verification_results']['pricing_info'] = pricing_info
                elif tool_name == 'verify_location':
                    location_info = self._verify_location(
                        tool_args.get('vendor_name'),
                        tool_args.get('address'),
                        tool_args.get('coordinates')
                    )
                    results['verification_results']['location_info'] = location_info
                elif tool_name == 'check_operating_hours':
                    hours_info = self._check_operating_hours(
                        tool_args.get('vendor_name'),
                        tool_args.get('date'),
                        tool_args.get('time')
                    )
                    results['verification_results']['operating_hours'] = hours_info

                results['tool_calls'].append({
                    'tool_name': tool_name,
                    'arguments': tool_args,
                    'result': results['verification_results'].get(f'{tool_name}_info', {})
                })

            # Calculate confidence score based on verification results
            results['confidence_score'] = self._calculate_verification_confidence(results['verification_results'])

            return results

        except Exception as e:
            logger.error(f"Error in online verification call: {e}")
            return {}

    def _get_verification_tools(self) -> List[Dict[str, Any]]:
        """Define the tools available for online verification"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_vendor_info",
                    "description": "Search for vendor information including business details, ratings, and reviews",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vendor_name": {
                                "type": "string",
                                "description": "Name of the vendor to search for"
                            }
                        },
                        "required": ["vendor_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_pricing",
                    "description": "Check pricing information for a specific service or product",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vendor_name": {
                                "type": "string",
                                "description": "Name of the vendor"
                            },
                            "service_type": {
                                "type": "string",
                                "description": "Type of service or product"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location where the service was provided"
                            },
                            "date": {
                                "type": "string",
                                "description": "Date of the service"
                            }
                        },
                        "required": ["vendor_name", "service_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_location",
                    "description": "Verify if a vendor exists at the specified location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vendor_name": {
                                "type": "string",
                                "description": "Name of the vendor"
                            },
                            "address": {
                                "type": "string",
                                "description": "Address to verify"
                            },
                            "coordinates": {
                                "type": "object",
                                "properties": {
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"}
                                },
                                "description": "Geographic coordinates"
                            }
                        },
                        "required": ["vendor_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_operating_hours",
                    "description": "Check if a vendor was operating at the specified date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vendor_name": {
                                "type": "string",
                                "description": "Name of the vendor"
                            },
                            "date": {
                                "type": "string",
                                "description": "Date to check"
                            },
                            "time": {
                                "type": "string",
                                "description": "Time to check"
                            }
                        },
                        "required": ["vendor_name", "date"]
                    }
                }
            }
        ]

    def _search_vendor_info(self, vendor_name: str) -> Dict[str, Any]:
        """Search for vendor information online"""
        try:
            # Implement actual vendor search logic here
            # This could use various APIs like Google Places, Yelp, etc.
            return {
                'vendor_exists': True,
                'business_type': 'Restaurant',  # Example
                'rating': 4.5,  # Example
                'reviews_count': 100,  # Example
                'verified': True  # Example
            }
        except Exception as e:
            logger.error(f"Error searching vendor info: {e}")
            return {}

    def _check_pricing(self, vendor_name: str, service_type: str, 
                      location: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """Check pricing information for a service"""
        try:
            # Implement actual pricing check logic here
            # This could use various APIs or web scraping
            return {
                'price_range': {
                    'min': 50.0,  # Example
                    'max': 100.0,  # Example
                    'average': 75.0  # Example
                },
                'currency': 'USD',
                'last_updated': datetime.now().isoformat(),
                'source': 'Online API'  # Example
            }
        except Exception as e:
            logger.error(f"Error checking pricing: {e}")
            return {}

    def _verify_location(self, vendor_name: str, address: Optional[str] = None,
                        coordinates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Verify vendor location"""
        try:
            # Implement actual location verification logic here
            # This could use Google Maps API, etc.
            return {
                'location_verified': True,
                'address_match': True if address else None,
                'coordinates_match': True if coordinates else None,
                'distance': 0.1  # Example: distance in km
            }
        except Exception as e:
            logger.error(f"Error verifying location: {e}")
            return {}

    def _check_operating_hours(self, vendor_name: str, date: str, time: Optional[str] = None) -> Dict[str, Any]:
        """Check if vendor was operating at specified date/time"""
        try:
            # Implement actual operating hours check logic here
            # This could use Google Places API, etc.
            return {
                'was_open': True,
                'operating_hours': {
                    'open': '09:00',
                    'close': '22:00'
                },
                'special_hours': False,
                'holiday': False
            }
        except Exception as e:
            logger.error(f"Error checking operating hours: {e}")
            return {}

    def _calculate_verification_confidence(self, verification_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on verification results"""
        try:
            confidence_factors = []
            
            # Check vendor verification
            if 'vendor_info' in verification_results:
                vendor_info = verification_results['vendor_info']
                if vendor_info.get('vendor_exists'):
                    confidence_factors.append(1.0)
                if vendor_info.get('verified'):
                    confidence_factors.append(0.8)
            
            # Check pricing verification
            if 'pricing_info' in verification_results:
                pricing_info = verification_results['pricing_info']
                if pricing_info.get('price_range'):
                    confidence_factors.append(0.9)
                if pricing_info.get('last_updated'):
                    confidence_factors.append(0.7)
            
            # Check location verification
            if 'location_info' in verification_results:
                location_info = verification_results['location_info']
                if location_info.get('location_verified'):
                    confidence_factors.append(1.0)
                if location_info.get('address_match'):
                    confidence_factors.append(0.8)
            
            # Check operating hours verification
            if 'operating_hours' in verification_results:
                hours_info = verification_results['operating_hours']
                if hours_info.get('was_open'):
                    confidence_factors.append(0.9)
                if not hours_info.get('special_hours'):
                    confidence_factors.append(0.7)
            
            # Return average confidence score
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating verification confidence: {e}")
            return 0.0

    def _extract_risk_factors_from_verification(self, verification_results: Dict[str, Any], category: str) -> List[str]:
        """Extract risk factors from verification results"""
        risk_factors = []
        
        try:
            # Check vendor verification
            if 'vendor_info' in verification_results:
                vendor_info = verification_results['vendor_info']
                if not vendor_info.get('vendor_exists'):
                    risk_factors.append(f"Vendor {category} not found in online databases")
                if not vendor_info.get('verified'):
                    risk_factors.append(f"Vendor {category} verification status unclear")
            
            # Check pricing verification
            if 'pricing_info' in verification_results:
                pricing_info = verification_results['pricing_info']
                if pricing_info.get('price_range'):
                    price_range = pricing_info['price_range']
                    if price_range.get('min') and price_range.get('max'):
                        risk_factors.append(f"Price outside typical range for {category}")
            
            # Check location verification
            if 'location_info' in verification_results:
                location_info = verification_results['location_info']
                if not location_info.get('location_verified'):
                    risk_factors.append(f"Location verification failed for {category}")
                if location_info.get('distance', 0) > 1.0:  # More than 1km away
                    risk_factors.append(f"Location significantly different from expected for {category}")
            
            # Check operating hours
            if 'operating_hours' in verification_results:
                hours_info = verification_results['operating_hours']
                if not hours_info.get('was_open'):
                    risk_factors.append(f"{category} was not operating at the specified time")
                if hours_info.get('special_hours'):
                    risk_factors.append(f"Transaction occurred during special hours for {category}")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error extracting risk factors: {e}")
            return []

    def _generate_summary(self, overall_risk_score: float, fraud_probability: float) -> str:
        """Generate a human-readable summary of the fraud analysis results"""
        try:
            # Calculate percentage for readability
            risk_percentage = int(overall_risk_score * 100)
            fraud_percentage = int(fraud_probability * 100)
            
            # Start with overall assessment
            if fraud_percentage >= 80:
                assessment = "HIGH RISK"
            elif fraud_percentage >= 50:
                assessment = "MODERATE RISK"
            else:
                assessment = "LOW RISK"
            
            summary = f"Fraud Analysis Summary:\n"
            summary += f"Overall Risk Assessment: {assessment}\n"
            summary += f"Fraud Probability: {fraud_percentage}%\n"
            summary += f"Risk Score: {risk_percentage}%\n\n"
            
            # Add key risk factors
            if self.risk_factors:
                summary += "Key Risk Factors:\n"
                for factor in self.risk_factors[:3]:  # Show top 3 risk factors
                    summary += f"- {factor}\n"
            
            # Add verification highlights
            if self.verification_results:
                summary += "\nVerification Highlights:\n"
                if 'inconsistent_dates' in self.verification_results:
                    date_info = self.verification_results['inconsistent_dates']
                    if date_info.get('date_mismatch'):
                        summary += f"- Date mismatch detected (Email: {date_info.get('email_date')}, Receipt: {date_info.get('receipt_date')})\n"
                
                if 'amount_verification' in self.verification_results:
                    amount_info = self.verification_results['amount_verification']
                    if not amount_info.get('amount_match'):
                        summary += f"- Amount mismatch detected (Expected: {amount_info.get('expense_data_amount')}, Receipt: {amount_info.get('receipt_amount')})\n"
            
            # Add image analysis highlights
            if self.image_analysis_results:
                if 'manipulation_indicators' in self.image_analysis_results:
                    indicators = self.image_analysis_results['manipulation_indicators']
                    if indicators:
                        summary += "\nImage Analysis:\n"
                        for indicator in indicators[:2]:  # Show top 2 manipulation indicators
                            summary += f"- {indicator}\n"
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary"

async def check_receipt_fraud(expense_id: UUID, file_url: str) -> Dict[str, Any]:
    """
    Check a receipt for potential fraud.
    
    Args:
        expense_id: UUID of the expense record
        file_url: URL of the receipt image/document
        
    Returns:
        Dictionary containing fraud analysis results
    """
    detector = ReceiptFraudDetector()
    return await detector.analyze_receipt(expense_id, file_url) 