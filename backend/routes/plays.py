from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from db import supabase
from embeddings import create_play_embedding, create_query_embedding

router = APIRouter(prefix="/plays")

class PlayBase(BaseModel):
    map: str
    agent: str
    enemy_agent: Optional[str]=None
    play_description: str

# when user creates a play
class PlayCreate(PlayBase):
    playbook_id: str

# when user updates a play
class PlayUpdate(BaseModel):
    map: Optional[str]=None
    agent: Optional[str]=None
    enemy_agent: Optional[str]=None
    play_description: Optional[str]=None

# response from database/need to tell pydantic to read it properly
class PlayResponse(PlayBase):
    id: str
    playbook_id: str
    created_at: str
    updated_at: Optional[str]=None
    user_id: Optional[str]=None
    class Config:
        from_attributes=True

# response from database for similar searches
class PlaySearchResponse(PlayResponse):
    similarity: float

# endpoints

# get plays based on filtering
@router.get("/", response_model=List[PlayResponse])
async def get_plays(map: Optional[str]=None, agent: Optional[str]=None, skip: int=0, limit: int=100):
    try:
        query = supabase.table("plays").select("*")
        if map:
            query = query.eq("map", map)
        if agent:
            query = query.eq("agent", agent)
        result = query.range(skip, skip + limit - 1).execute()
        return result.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching plays: {str(e)}"
        )

# create play in supabase with embedding
@router.post("/", response_model=PlayResponse, status_code=status.HTTP_201_CREATED)
async def create_play(play: PlayCreate):
    try:
        # create embedding
        embedding = create_play_embedding({
            "map": play.map,
            "agent": play.agent,
            "enemy_agent": play.enemy_agent,
            "play_description": play.play_description
        })
        play_data = play.model_dump()
        play_data["embedding"] = embedding
        result = supabase.table("plays").insert(play_data).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating play: {str(e)}"
        )

# search for similar plays according to user query
@router.get("/search/similar", response_model=List[PlaySearchResponse], status_code=status.HTTP_200_OK)
async def get_similar_play(query: str, map: Optional[str]=None, agent: Optional[str]=None, enemy_agent: Optional[str]=None, threshold: float=0.7, limit: int=5):
    try:
        # create context dictionary for embedding
        context = {}
        if map:
            context["map"] = map
        if agent:
            context["agent"] = agent
        if enemy_agent:
            context["enemy_agent"] = enemy_agent

        # create embedding for the query
        query_embedding = create_query_embedding(query, context)

        # call supabase function
        response = supabase.rpc(
            "match_plays",
            {
                "query_embedding": query_embedding,
                "filter_map": map,
                "filter_agent": agent,
                "filter_enemy_agent": enemy_agent,
                "match_threshold": threshold,
                "match_count": limit
            }
        ).execute()
        return response.data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching for similar plays: {str(e)}"
        )

# delete play based on id
@router.delete("/{play_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_play(play_id: str):
    try:
        response = supabase.table("plays").delete().eq("id", play_id).execute()
        if not response.data:
             raise HTTPException(status_code=404, detail="Play not found")
        return None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting play: {str(e)}"
        )

# update play based on id
@router.put("/{play_id}", response_model=PlayResponse, status_code=status.HTTP_200_OK)
async def update_play(play_id: str, updated_play: PlayUpdate):
    try:
        # first get previous info
        current_play_response = supabase.table("plays").select("*").eq("id", play_id).execute()
        if not current_play_response.data:
            raise HTTPException(status_code=404, detail="Play not found")
        current_play = current_play_response.data[0]
        # get any incomming changes only if htere are
        incoming_changes = updated_play.model_dump(exclude_unset=True)
        if not incoming_changes:
            return current_play
        # merge data
        merged_data = {**current_play, **incoming_changes}
        # get new embedding for updated_play
        relevant_fields = ["map", "agent", "enemy_agent", "play_description"]
        if any(field in incoming_changes for field in relevant_fields):
             new_embedding = create_play_embedding({
                "map": merged_data["map"],
                "agent": merged_data["agent"],
                "enemy_agent": merged_data.get("enemy_agent"),
                "play_description": merged_data["play_description"]
             })
             incoming_changes["embedding"] = new_embedding

        result = supabase.table("plays").update(incoming_changes).eq("id", play_id).execute()
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating play: {str(e)}"
        )