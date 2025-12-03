import random
from typing import Callable, List, Iterable, Dict, Optional, Union, Tuple

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from utils.distances import jaccard_sim

user_col = 'user_id'
item_col = 'item_id'
rating_col = 'rating'

def tqdm_or_identity(
    iterable: Iterable,
    verbose: bool=False,
) -> Callable[[Iterable], None]:
  if verbose:
    return tqdm(iterable)
  else:
    return iterable


similarityFuncType = Callable[[NDArray[float], NDArray[float]], NDArray[float]]

class BaseModel:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.n_users = len(np.unique(self.ratings['userId']))
        self.n_items = len(np.unique(self.ratings['trackId']))

        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.ratings['userId'], self.ratings['trackId']] = 1.
        
    def recommend(self, uid: int):
        """
        param uid: int - user's id
        return: [n_items] - vector of recommended items sorted by their scores in descending order
        """
        raise NotImplementedError

    def remove_train_items(self, preds: List[List[int]], k: int):
        """
        param preds: [n_users, n_items] - recommended items for each user
        param k: int
        return: np.array [n_users, k] - recommended items without training examples
        """
        new_preds = np.zeros((len(preds), k), dtype=int)
        for user_id, user_data in self.ratings.groupby('userId'):
            user_preds = preds[user_id]
            new_preds[user_id] = user_preds[~np.in1d(user_preds, user_data['trackId'])][:k]

        return new_preds

    def get_test_recommendations(self, test_idxs: List[int], k: int) -> NDArray[int]:
        # your code here
        pass
        

class RandomRecommender(BaseModel):
    def __init__(self, ratings):
        super().__init__(ratings)

    def recommend(self, uid: int):
        unique_items = self.ratings['trackId'].unique()
        predictions_u = np.random.permutation(unique_items)
        return predictions_u
    

class User2User(BaseModel):
    def __init__(self, ratings, similarity_func: similarityFuncType=jaccard_sim, alpha: float=0.02):
        super().__init__(ratings)

        self.similarity_func = similarity_func
        self.alpha = alpha

    def similarity(self, user_vector: NDArray[int]) -> NDArray[float]:
        """Computes similarities between user_vector and all vectors in self.R
        Args:
            user_vector: vector of ratings, user has given to all tracks
        Returns:
            vector of simillarities between this user and all users in self.R
        """
        # your code here:
        
        return self.similarity_func(self.R, user_vector)
    
    def get_items_scores(self, uid: int) -> NDArray[float]:
        """Computes scores \hat{r}_{ui} (https://www.overleaf.com/read/vcqdvjkbjgzf#9c30ef) for all items in rating matrix for
        a particular user uid. Don't forget, that similarity of user u with himself should
        be zero to not get recommendetations from his own interactions.

        Args:
            uid (int): index of user from rating matrix
        Returns:
            scores_u (NDArray[float]): array of scores for all items
        """
        # your code here:

        #self.R = [
        #[1, 0, 1, 0, ...],  # Пользователь 0: 1 = слушал трек, 0 = не слушал
        #[0, 1, 0, 1, ...],  # Пользователь 1
        #[1, 0, 0, 1, ...],  # Пользователь 2
        #...

        #Посчитаем вектор в котором будут меры Жаккара.
        jaccard = self.similarity(self.R[uid])
        jaccard[uid]=0 #Схожесть Жаккара нашего объекта uid с самим собой равна нулю по условию.

        #Определим множество соседей.

        jaccard_neighbours={} #{index_usera: мера Жаккара}

        for i in range(0,len(jaccard)):
            if jaccard[i]>self.alpha:
                jaccard_neighbours[i]=jaccard[i]
        
        #Знаменатель формулы
        znamenatel=0.0
        for i in jaccard_neighbours:
            znamenatel+=jaccard_neighbours[i]

        #Числитель для формулы

        chislitel=np.zeros(self.R.shape[1])
        r=np.zeros(self.R.shape[1])
        for j in range(0,self.R.shape[1],1): #Бежим по всем трекам
            for i in jaccard_neighbours: #По всем соседям идем
                chislitel[j]+=jaccard_neighbours[i]*self.R[i,j]
            if znamenatel!=0:
                r[j]=chislitel[j]/znamenatel
            else:
                r[j]=0


        return r


    

        


    def recommend(self, uid: int):
        scores_u = self.get_items_scores(uid=uid)
        predictions_u = np.array([idx for idx in np.argsort(scores_u)[::-1]])
        return predictions_u
    
    

def _als_user_step(
    items_embeddings: NDArray[float],
    user_ratings: NDArray[float],
    reg_coef: float,
) -> NDArray[float]:
    
    # КРИТИЧЕСКИ ВАЖНО: принудительная конвертация
    V = np.asarray(items_embeddings, dtype=np.float64).copy()
    r = np.asarray(user_ratings, dtype=np.float64).copy()
    
    # Проверка
    if not np.isfinite(V).all() or not np.isfinite(r).all():
        raise ValueError("Матрица содержит нечисловые значения")
    
    emb_dim = V.shape[1]
    
    # Вычисления
    V_t = V.T
    A = V_t @ V + float(reg_coef) * np.eye(emb_dim, dtype=np.float64)
    
    # inv должен работать теперь
    A_inv = np.linalg.inv(A)
    b = V_t @ r
    
    return A_inv @ b

def _als_item_step(
    users_embeddings: NDArray[float],
    items_ratings: NDArray[float],
    reg_coef: float,
) -> NDArray[float]:
    """
    ALS model (https://yadi.sk/i/7ZONA2kIqROfRQ) consists of 2 steps: 
        1) recompute users embeddings
        2) recompute items embeddings
    
    This function allows one to recompute embedding for one particular item,
    given ratings that we have for this item from different users and users_embeddings of those users, who have rated this item
    """
    # your code here
    
    lamb=float(reg_coef)
    r=items_ratings.astype(float)
    U=users_embeddings.astype(float)
    U_t=U.T
    UU=U_t @ U
    dim=UU.shape[0]
    I=np.eye(dim)

    v = np.linalg.inv(U_t @ U + lamb*I)@U_t@r

    return v
    

UserID = Union[str, int]
ItemID = Union[str, int]
UserIDs = List[UserID]
ItemIDs = List[ItemID]
RatingsType = List[float]
EmbeddingType = Dict[Union[UserID, ItemID], List[float]]
"""
{
    "u1": [1.0, -1.3, 3.5, 4.1],
    "u2": [-4.5, 3.3, 3.5, 0.1],
    "u3": [-4.65, -1.8, 4.18, 2.12],
    ...
}
"""
UserInfoType = Dict[UserID, Tuple[ItemIDs, RatingsType]]
"""
{
    "u1": (["i1", "i4", ...], [5.0, 3.5, ...]),
    ...
}
"""
ItemInfoType = Dict[ItemID, Tuple[UserIDs, RatingsType]]
"""
{
    "i1": (["u3", "u5", ...], [1.5, 5.0, ...]),
    ...
}
"""

class ALS:
    """
    Class for training LFM model with ALS algorithm
    (formula (1.2) from this file: https://yadi.sk/i/7ZONA2kIqROfRQ)
    NOTE: in this inplementation we won't train user and item biases, thus the optimisation
        problem looks like that:
        
        ∑_{(u,i): \exists r_{ui}} (r_{ui} - <p_{u}, q_{i}>)^2 + reg_coef/2 * (∑_{u} ||p_{u}||^2 + ∑_{i} ||q_{i}||^2)  -> min_{P, Q}
    """

    def __init__(
        self,
        embeddings_dim: int=16,
        reg_coef: float=1.0,
        random_seed: int=59812
    ):
        self.embeddings_dim = embeddings_dim
        self.random_seed = random_seed
        self.reg_coef = reg_coef
        self.user2emb, self.item2emb = {}, {}
        self.users_embeddings: Optional[EmbeddingType] = None
        self.items_embeddings: Optional[EmbeddingType] = None
        
    def fit(
        self, 
        interactions: pd.DataFrame, 
        epochs: int=20, 
        verbose: bool=False,
        embeddings_initialized: bool=False,
    ):
        """
        Trains the model - iteratively recomputes self.users_embeddings and self.items_embeddings
        
        Args:
            interactions: dataframe of interactions (necessary columns: user_col, item_col, rating_col)
                for model training
            epochs: amount of iterations to recompute users_embeddings and items_embeddings
            verbose: whether to do additional logging during training or not
            embeddings_initialized (bool): whether to initialize self.users_embeddings and self.items_embeddings or
                they are already initialized
                
        IMPORTANT NOTE: first recompute users embeddings, then items embeddings
        """
        # your code here
        return self

