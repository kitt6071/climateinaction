import re
import logging

logger = logging.getLogger(__name__)

class EcologicalKnowledgeProcessor:
    def __init__(self):
        self.iucn_threats = self._initialize_iucn_classification()
        self.iucn_stresses = self._initialize_stress_classification()
        self.impact_patterns = self._initialize_impact_patterns()
        
    def _initialize_iucn_classification(self):
        return {
            '1': {'category': 'Residential & commercial development', 'subcategories': {
                '1.1': 'Housing & urban areas',
                '1.2': 'Commercial & industrial areas',
                '1.3': 'Tourism & recreation areas'
            }},
            '2': {'category': 'Agriculture & aquaculture', 'subcategories': {
                '2.1': 'Annual & perennial non-timber crops',
                '2.2': 'Wood & pulp plantations',
                '2.3': 'Livestock farming & ranching',
                '2.4': 'Marine & freshwater aquaculture'
            }},
            '3': {'category': 'Energy production & mining', 'subcategories': {
                '3.1': 'Oil & gas drilling',
                '3.2': 'Mining & quarrying',
                '3.3': 'Renewable energy'
            }},
            '4': {'category': 'Transportation & service corridors', 'subcategories': {
                '4.1': 'Roads & railroads',
                '4.2': 'Utility & service lines',
                '4.3': 'Shipping lanes',
                '4.4': 'Flight paths'
            }},
            '5': {'category': 'Biological resource use', 'subcategories': {
                '5.1': 'Hunting & collecting terrestrial animals',
                '5.2': 'Gathering terrestrial plants',
                '5.3': 'Logging & wood harvesting',
                '5.4': 'Fishing & harvesting aquatic resources'
            }},
            '6': {'category': 'Human intrusions & disturbance', 'subcategories': {
                '6.1': 'Recreational activities',
                '6.2': 'War, civil unrest & military exercises',
                '6.3': 'Work & other activities'
            }},
            '7': {'category': 'Natural system modifications', 'subcategories': {
                '7.1': 'Fire & fire suppression',
                '7.2': 'Dams & water management/use',
                '7.3': 'Other ecosystem modifications'
            }},
            '8': {'category': 'Invasive & other problematic species', 'subcategories': {
                '8.1': 'Invasive non-native/alien species/diseases',
                '8.2': 'Problematic native species/diseases',
                '8.3': 'Introduced genetic material',
                '8.4': 'Problematic species/diseases of unknown origin',
                '8.5': 'Viral/prion-induced diseases',
                '8.6': 'Diseases of unknown cause'
            }},
            '9': {'category': 'Pollution', 'subcategories': {
                '9.1': 'Domestic & urban waste water',
                '9.2': 'Industrial & military effluents',
                '9.3': 'Agricultural & forestry effluents',
                '9.4': 'Garbage & solid waste',
                '9.5': 'Air-borne pollutants',
                '9.6': 'Excess energy'
            }},
            '10': {'category': 'Geological events', 'subcategories': {
                '10.1': 'Volcanoes',
                '10.2': 'Earthquakes/tsunamis',
                '10.3': 'Avalanches/landslides'
            }},
            '11': {'category': 'Climate change & severe weather', 'subcategories': {
                '11.1': 'Habitat shifting & alteration',
                '11.2': 'Droughts',
                '11.3': 'Temperature extremes',
                '11.4': 'Storms & flooding',
                '11.5': 'Other impacts'
            }},
            '12': {'category': 'Other options', 'subcategories': {
                '12.1': 'Other threat'
            }}
        }
    
    def _initialize_stress_classification(self):
        return {
            'ecosystem_conversion': 'Complete habitat loss',
            'ecosystem_degradation': 'Habitat quality decline',
            'indirect_ecosystem_effects': 'Secondary habitat impacts',
            'species_mortality': 'Direct species killing',
            'species_disturbance': 'Behavioral disruption',
            'reduced_reproductive_success': 'Breeding impacts',
            'reduced_recruit_survival': 'Juvenile survival',
            'competition': 'Interspecific competition',
            'predation': 'Predation pressure',
            'poisoning': 'Toxic exposure',
            'disease': 'Pathogenic impacts',
            'genetic_effects': 'Genetic diversity loss',
            'hybridization': 'Genetic pollution'
        }
    
    def _initialize_impact_patterns(self):
        return {
            'magnitude_patterns': [
                r'\b(severe|major|significant|substantial|extensive|massive|dramatic)\b',
                r'\b(moderate|limited|minor|slight|small|reduced)\b',
                r'\b(complete|total|entire|whole|full)\b',
                r'\b(partial|some|certain|specific)\b'
            ],
            'causality_patterns': [
                r'\b(cause[sd]?|lead[s]? to|result[s]? in|trigger[s]?|induce[s]?)\b',
                r'\b(due to|because of|as a result of|owing to)\b',
                r'\b(contribute[s]? to|influence[s]?|affect[s]?|impact[s]?)\b'
            ],
            'temporal_patterns': [
                r'\b(immediate|instant|rapid|quick|sudden)\b',
                r'\b(gradual|slow|progressive|chronic|long-term)\b',
                r'\b(annual|seasonal|periodic|cyclic)\b',
                r'\b(historic|past|recent|current|ongoing)\b'
            ],
            'directness_patterns': [
                r'\b(direct[ly]?|immediate[ly]?|straight|explicit)\b',
                r'\b(indirect[ly]?|secondary|consequent|mediated)\b',
                r'\b(cascading|knock-on|ripple|downstream)\b'
            ]
        }

    def classify_threat_to_iucn(self, threat_text):
        threat_lower = threat_text.lower()
        
        category_keywords = {
            '1': ['urban', 'housing', 'development', 'commercial', 'industrial', 'tourism', 'recreation'],
            '2': ['agriculture', 'farming', 'livestock', 'aquaculture', 'plantation', 'crops'],
            '3': ['mining', 'oil', 'gas', 'drilling', 'renewable', 'energy', 'quarrying'],
            '4': ['road', 'railroad', 'transport', 'shipping', 'utility', 'corridor'],
            '5': ['hunting', 'fishing', 'harvesting', 'logging', 'collecting', 'exploitation'],
            '6': ['recreation', 'disturbance', 'human', 'war', 'military'],
            '7': ['fire', 'dam', 'water management', 'ecosystem modification'],
            '8': ['invasive', 'alien', 'disease', 'pathogen', 'introduced'],
            '9': ['pollution', 'contamination', 'waste', 'chemical', 'toxic'],
            '10': ['volcano', 'earthquake', 'landslide', 'geological'],
            '11': ['climate', 'temperature', 'weather', 'drought', 'storm', 'flood'],
            '12': ['other', 'unknown', 'unspecified']
        }
        
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in threat_lower)
            if score > best_score:
                best_score = score
                best_match = category
        
        confidence = min(best_score / 3.0, 1.0)
        
        return {
            'category': best_match,
            'confidence': confidence,
            'category_name': self.iucn_threats.get(best_match, {}).get('category', 'Unknown') if best_match else 'Unclassified'
        }

    def analyze_impact_statement(self, statement):
        if not statement:
            return {}
        
        statement_lower = statement.lower()
        
        magnitude = self._extract_magnitude(statement)
        
        causality = self._extract_causality(statement)
        
        temporality = self._extract_temporality(statement)
        
        directness = self._extract_directness(statement)
        
        mechanisms = self._extract_mechanisms(statement)
        
        confidence = self._calculate_analysis_confidence(statement)
        
        impact_outcomes = self._extract_impact_outcomes(statement)
        
        return {
            'magnitude': magnitude,
            'causality': causality,
            'temporality': temporality,
            'directness': directness,
            'mechanisms': mechanisms,
            'impact_outcomes': impact_outcomes,
            'confidence': confidence,
            'iucn_classification': self.classify_threat_to_iucn(statement),
            'processed_statement': statement
        }
    
    def _extract_impact_outcomes(self, text):
        text_lower = text.lower()
        outcomes = []
        
        outcome_patterns = {
            'population_decline': ['population decline', 'population decrease', 'population reduction', 'decline in population'],
            'mortality': ['mortality', 'death', 'killing', 'die', 'died', 'dies'],
            'habitat_loss': ['habitat loss', 'habitat destruction', 'habitat degradation', 'loss of habitat'],
            'breeding_failure': ['breeding failure', 'nesting failure', 'reproduction failure', 'failed breeding'],
            'displacement': ['displacement', 'forced migration', 'abandonment', 'relocate'],
            'stress': ['stress', 'physiological stress', 'behavioral stress'],
            'reduced_fitness': ['reduced fitness', 'fitness decline', 'lower fitness'],
            'extinction': ['extinction', 'extirpation', 'local extinction']
        }
        
        for outcome_type, patterns in outcome_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    outcomes.append({
                        'type': outcome_type,
                        'pattern': pattern,
                        'confidence': 0.8
                    })
                    break
        
        return outcomes
    
    def _extract_magnitude(self, text):
        high_magnitude = ['severe', 'major', 'significant', 'substantial', 'extensive', 'massive', 'dramatic', 'complete', 'total']
        medium_magnitude = ['moderate', 'noticeable', 'considerable', 'partial']
        low_magnitude = ['minor', 'slight', 'small', 'limited', 'minimal']
        
        for word in high_magnitude:
            if word in text:
                return {'level': 'high', 'indicators': [word]}
        for word in medium_magnitude:
            if word in text:
                return {'level': 'medium', 'indicators': [word]}
        for word in low_magnitude:
            if word in text:
                return {'level': 'low', 'indicators': [word]}
        
        return {'level': 'unknown', 'indicators': []}
    
    def _extract_causality(self, text):
        strong_causal = ['cause', 'lead to', 'result in', 'trigger', 'induce']
        weak_causal = ['contribute to', 'influence', 'affect', 'impact']
        
        for phrase in strong_causal:
            if phrase in text:
                return {'strength': 'strong', 'indicators': [phrase]}
        for phrase in weak_causal:
            if phrase in text:
                return {'strength': 'weak', 'indicators': [phrase]}
        
        return {'strength': 'unknown', 'indicators': []}
    
    def _extract_temporality(self, text):
        immediate = ['immediate', 'instant', 'rapid', 'quick', 'sudden']
        gradual = ['gradual', 'slow', 'progressive', 'chronic', 'long-term']
        periodic = ['annual', 'seasonal', 'periodic', 'cyclic']
        
        for word in immediate:
            if word in text:
                return {'pattern': 'immediate', 'indicators': [word]}
        for word in gradual:
            if word in text:
                return {'pattern': 'gradual', 'indicators': [word]}
        for word in periodic:
            if word in text:
                return {'pattern': 'periodic', 'indicators': [word]}
        
        return {'pattern': 'unknown', 'indicators': []}
    
    def _extract_directness(self, text):
        direct_indicators = ['direct', 'immediate', 'straight', 'explicit']
        indirect_indicators = ['indirect', 'secondary', 'consequent', 'mediated', 'cascading', 'knock-on', 'ripple', 'downstream']
        
        for word in direct_indicators:
            if word in text:
                return {'type': 'direct', 'confidence': 0.8, 'indicators': [word]}
        for word in indirect_indicators:
            if word in text:
                return {'type': 'indirect', 'confidence': 0.8, 'indicators': [word]}
        
        if any(phrase in text for phrase in ['kill', 'death', 'mortality', 'destroy']):
            return {'type': 'direct', 'confidence': 0.6, 'indicators': ['implicit']}
        
        return {'type': 'ambiguous', 'confidence': 0.3, 'indicators': []}
    
    def _extract_mechanisms(self, text):
        mechanisms = {
            'habitat_loss': ['habitat loss', 'deforestation', 'destruction', 'clearance'],
            'pollution': ['pollution', 'contamination', 'chemical', 'toxic', 'pesticide'],
            'climate_change': ['temperature', 'warming', 'climate', 'weather', 'precipitation'],
            'disease': ['disease', 'pathogen', 'virus', 'infection', 'parasite'],
            'competition': ['competition', 'compete', 'displacement', 'outcompete'],
            'predation': ['predation', 'predator', 'prey', 'hunting', 'consumption'],
            'disturbance': ['disturbance', 'noise', 'light', 'traffic', 'human activity']
        }
        
        identified_mechanisms = []
        for mechanism, keywords in mechanisms.items():
            if any(keyword in text for keyword in keywords):
                identified_mechanisms.append(mechanism)
        
        return identified_mechanisms
    
    def _calculate_analysis_confidence(self, statement):
        if not statement:
            return 0.0
        
        confidence_factors = [
            len(statement.split()) > 5,
            any(word in statement.lower() for word in ['study', 'research', 'observed', 'measured']),
            bool(re.search(r'\d+', statement)),
            any(word in statement.lower() for word in ['significant', 'p <', 'correlation', 'analysis'])
        ]
        
        base_confidence = 0.3
        bonus = sum(confidence_factors) * 0.15
        
        return min(base_confidence + bonus, 1.0) 