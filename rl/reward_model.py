"""
Reward Model for RL Chat System

This module implements reward scoring for chat responses to enable
reinforcement learning. It provides both simple rating-based rewards
and more sophisticated multi-criteria scoring.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class RewardModel(ABC):
    """Abstract base class for reward models."""
    
    @abstractmethod
    def score_response(self, user_message: str, response: str, 
                      feedback: Optional[Dict] = None) -> float:
        """
        Score a response and return a reward signal.
        
        Args:
            user_message: The user's input message
            response: The model's generated response
            feedback: Optional user feedback (ratings, corrections, etc.)
            
        Returns:
            Reward score between 0.0 and 1.0 (higher is better)
        """
        pass
    
    @abstractmethod
    def update_from_feedback(self, feedback_data: List[Dict]) -> None:
        """
        Update the reward model based on collected feedback.
        
        Args:
            feedback_data: List of feedback examples for learning
        """
        pass


class SimpleRatingReward(RewardModel):
    """
    Simple reward model based on user ratings.
    
    This is the most basic reward model that uses direct user feedback
    (thumbs up/down) to score responses.
    """
    
    def __init__(self, default_reward: float = 0.5):
        """
        Initialize the simple rating reward model.
        
        Args:
            default_reward: Default reward for unrated responses
        """
        self.default_reward = default_reward
        self.rating_history = []
        
    def score_response(self, user_message: str, response: str, 
                      feedback: Optional[Dict] = None) -> float:
        """
        Score response based on user rating.
        
        Args:
            user_message: The user's input message
            response: The model's generated response  
            feedback: Dict with 'rating' key (1 for positive, 0 for negative)
            
        Returns:
            Reward score (1.0 for positive rating, 0.0 for negative, default for no rating)
        """
        if feedback is None or 'rating' not in feedback:
            logger.debug("No rating feedback provided, using default reward")
            return self.default_reward
            
        rating = feedback['rating']
        if rating == 1:
            reward = 1.0
        elif rating == 0:
            reward = 0.0
        else:
            logger.warning(f"Invalid rating {rating}, using default reward")
            reward = self.default_reward
            
        # Store for learning
        self.rating_history.append({
            'user_message': user_message,
            'response': response,
            'rating': rating,
            'reward': reward
        })
        
        logger.debug(f"Scored response: rating={rating}, reward={reward}")
        return reward
    
    def update_from_feedback(self, feedback_data: List[Dict]) -> None:
        """
        Update model based on feedback data.
        
        For simple rating model, this just stores the feedback.
        More sophisticated models could learn patterns from this data.
        
        Args:
            feedback_data: List of feedback examples
        """
        self.rating_history.extend(feedback_data)
        logger.info(f"Updated reward model with {len(feedback_data)} feedback examples")
        logger.info(f"Total feedback examples: {len(self.rating_history)}")


class MultiCriteriaReward(RewardModel):
    """
    Multi-criteria reward model that scores responses on multiple dimensions.
    
    This model evaluates responses on criteria like relevance, helpfulness,
    safety, and coherence, then combines them into a single reward score.
    """
    
    def __init__(self, criteria_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-criteria reward model.
        
        Args:
            criteria_weights: Weights for different criteria (must sum to 1.0)
        """
        self.criteria_weights = criteria_weights or {
            'relevance': 0.3,
            'helpfulness': 0.3, 
            'safety': 0.2,
            'coherence': 0.2
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.criteria_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")
            
        self.feedback_history = []
        
    def score_response(self, user_message: str, response: str,
                      feedback: Optional[Dict] = None) -> float:
        """
        Score response on multiple criteria.
        
        Args:
            user_message: The user's input message
            response: The model's generated response
            feedback: Dict with criteria scores or overall rating
            
        Returns:
            Weighted reward score between 0.0 and 1.0
        """
        if feedback is None:
            # No feedback, return neutral score
            return 0.5
            
        # Extract criteria scores from feedback
        criteria_scores = {}
        for criterion in self.criteria_weights.keys():
            if criterion in feedback:
                criteria_scores[criterion] = feedback[criterion]
            else:
                # If specific criterion not provided, use overall rating
                criteria_scores[criterion] = feedback.get('rating', 0.5)
                
        # Calculate weighted score
        weighted_score = sum(
            criteria_scores[criterion] * weight
            for criterion, weight in self.criteria_weights.items()
        )
        
        # Store for learning
        self.feedback_history.append({
            'user_message': user_message,
            'response': response,
            'criteria_scores': criteria_scores,
            'weighted_score': weighted_score,
            'feedback': feedback
        })
        
        logger.debug(f"Multi-criteria score: {criteria_scores} -> {weighted_score:.3f}")
        return weighted_score
    
    def update_from_feedback(self, feedback_data: List[Dict]) -> None:
        """
        Update model based on feedback data.
        
        This could implement learning algorithms to improve scoring
        based on user feedback patterns.
        
        Args:
            feedback_data: List of feedback examples
        """
        self.feedback_history.extend(feedback_data)
        logger.info(f"Updated multi-criteria model with {len(feedback_data)} examples")
        
        # TODO: Implement learning from feedback patterns
        # This could include:
        # - Learning user preferences
        # - Adjusting criteria weights
        # - Detecting feedback patterns


class LearnedRewardModel(RewardModel):
    """
    Learned reward model that adapts based on user feedback.
    
    This is a placeholder for a more sophisticated model that could
    learn to predict user satisfaction from conversation context.
    """
    
    def __init__(self):
        """Initialize the learned reward model."""
        self.model = None  # Placeholder for actual ML model
        self.training_data = []
        
    def score_response(self, user_message: str, response: str,
                      feedback: Optional[Dict] = None) -> float:
        """
        Score response using learned model.
        
        Args:
            user_message: The user's input message
            response: The model's generated response
            feedback: Optional user feedback
            
        Returns:
            Predicted reward score
        """
        if self.model is None:
            # No model trained yet, use simple heuristic
            return 0.5
            
        # TODO: Implement actual model prediction
        # This would involve:
        # - Encoding user message and response
        # - Running through trained model
        # - Returning predicted reward
        
        return 0.5
    
    def update_from_feedback(self, feedback_data: List[Dict]) -> None:
        """
        Update the learned model with new feedback.
        
        Args:
            feedback_data: List of feedback examples for training
        """
        self.training_data.extend(feedback_data)
        logger.info(f"Added {len(feedback_data)} examples to training data")
        
        # TODO: Implement model training
        # This would involve:
        # - Preparing training data
        # - Training/updating the model
        # - Validating performance


def create_reward_model(model_type: str = "simple", **kwargs) -> RewardModel:
    """
    Factory function to create reward models.
    
    Args:
        model_type: Type of reward model ("simple", "multi_criteria", "learned")
        **kwargs: Additional arguments for the specific model type
        
    Returns:
        Initialized reward model
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "simple":
        return SimpleRatingReward(**kwargs)
    elif model_type == "multi_criteria":
        return MultiCriteriaReward(**kwargs)
    elif model_type == "learned":
        return LearnedRewardModel(**kwargs)
    else:
        raise ValueError(f"Unsupported reward model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test simple rating model
    print("Testing Simple Rating Reward Model:")
    simple_model = create_reward_model("simple")
    
    # Test with positive feedback
    reward = simple_model.score_response(
        "Hello, how are you?",
        "I'm doing well, thank you for asking!",
        {"rating": 1}
    )
    print(f"Positive feedback reward: {reward}")
    
    # Test with negative feedback  
    reward = simple_model.score_response(
        "What's the weather like?",
        "I don't know about weather.",
        {"rating": 0}
    )
    print(f"Negative feedback reward: {reward}")
    
    # Test multi-criteria model
    print("\nTesting Multi-Criteria Reward Model:")
    multi_model = create_reward_model("multi_criteria")
    
    reward = multi_model.score_response(
        "How do I cook pasta?",
        "To cook pasta, bring water to boil, add salt, then add pasta and cook according to package directions.",
        {
            "relevance": 1.0,
            "helpfulness": 0.9,
            "safety": 1.0,
            "coherence": 0.8
        }
    )
    print(f"Multi-criteria reward: {reward:.3f}")
