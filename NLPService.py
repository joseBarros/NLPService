from docopt import Optional
from flask import Flask, request, jsonify
from future.backports.test.ssl_servers import threading
from pymongo import MongoClient
from bson import ObjectId, DBRef
import spacy
import os

from rpds import List

app = Flask(__name__)

class NLPService:
    def __init__(self, db_uri="mongodb://localhost:27017", db_name="geneticsCollab"):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.nlp_model_collection = self.db["nlp_model"]
        self.named_entity_collection = self.db["named_entity"]
        self.article_collection = self.db["article"]
        self.tag_collection = self.db['tag']
        self.nlp_model_collection = self.db['nlp_model']

        self.thread_local = threading.local()

    def get_spacy_model(self, model_path):
        if not hasattr(self.thread_local, 'spacy_model'):
            if not os.path.exists(model_path):
                raise ValueError(f"Model path '{model_path}' does not exist")
            self.thread_local.spacy_model = spacy.load(model_path)
        return self.thread_local.spacy_model


    def remove_old_named_entities(self, article_id):
        article = self.article_collection.find_one({"_id": ObjectId(article_id)})

        if article and 'entities' in article:
            for entity_ref in article['namedEntities']:
                self.named_entity_collection.delete_one({"_id": entity_ref.id})

        self.article_collection.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": {"namedEntities": []}}
        )

    def get_or_create_tag(self, label):
        tag = self.tag_collection.find_one({'label': label})
        if tag:
            return DBRef('tag', tag['_id'])
        else:
            tag_id = ObjectId()
            self.tag_collection.insert_one({'_id': tag_id, 'label': label})
            return DBRef('tag', tag_id)

    def save_named_entity(self, text, start_char, end_char, tags, article_id):
        named_entity_id = ObjectId()
        named_entity_data = {
            '_id': named_entity_id,
            'text': text,
            'start_char': start_char,
            'end_char': end_char,
            'tags': tags,  # Tags will be DBRefs,
            'article': DBRef('article', article_id)
        }
        self.named_entity_collection.insert_one(named_entity_data)
        return DBRef('named_entity', named_entity_id)

    def save_entities_to_article(self, article_id, entities):
        entity_dbrefs = []
        for ent in entities:
            tags = [self.get_or_create_tag(ent['label'])]
            entity_dbrefs.append(self.save_named_entity(ent['text'], ent['start'], ent['end'], tags, article_id))

        self.article_collection.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": {"namedEntities": entity_dbrefs}}  # Replace old entities with the new ones
        )

    def get_model_tags(self, model_id):
        nlp_model = self.nlp_model_collection.find_one({"_id": ObjectId(model_id)})

        if not nlp_model or 'tags' not in nlp_model:
            raise ValueError("Invalid model_id or no tags associated with the model")

        # Retrieve all tag labels associated with this model
        tag_refs = nlp_model.get('tags', [])
        tag_labels = []
        for tag_ref in tag_refs:
            tag = self.tag_collection.find_one({"_id": tag_ref.id})
            if tag and 'label' in tag:
                tag_labels.append(tag['label'])

        return tag_labels

    def add_tags_to_model(self, model_id, tags):
        """Add new tags to the model"""
        nlp_model = self.nlp_model_collection.find_one({"_id": ObjectId(model_id)})
        if not nlp_model:
            raise ValueError("Model not found")

        existing_tag_refs = nlp_model.get('tags', [])

        # Convert DBRefs to a set of ObjectId to check against
        existing_tag_ids = {tag_ref.id for tag_ref in existing_tag_refs}

        # Loop new tags and add them if not present
        new_tag_refs = []
        for tag_label in tags:
            # Create/retrieve the tag DBRef
            tag_ref = self.get_or_create_tag(tag_label)

            if tag_ref.id not in existing_tag_ids:
                new_tag_refs.append(tag_ref)

        # Update the model's tags
        if new_tag_refs:
            self.nlp_model_collection.update_one(
                {"_id": ObjectId(model_id)},
                {"$push": {"tags": {"$each": new_tag_refs}}}  # Append all new tags
            )

    def process_text_and_save_entities(self, article_id, text, model_id):
        nlp_model = self.nlp_model_collection.find_one({"_id": ObjectId(model_id)})

        if not nlp_model or 'path' not in nlp_model:
            raise ValueError("Invalid model_id or model path")

        model_path = nlp_model['path']
        if not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist")

        nlp = self.get_spacy_model(model_path)

        doc = nlp(text)

        ner_labels: Optional[List[str]] = None
        ner_labels = ner_labels or [label for label in nlp.get_pipe("ner").labels if label != "O"]
        print(ner_labels)

        # Filter entities based on allowed labels
        entities = [
            {'text': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_}
            for ent in doc.ents if ent.label_ in ner_labels
        ]
        print(entities)
        # Remove old named entities associated with the article
        self.remove_old_named_entities(article_id)

        # Save new entities and link to the article
        self.save_entities_to_article(article_id, entities)

        # Add new tags to the model
        new_tags = set(ent['label'] for ent in entities)  # Extract labels (tags) from entities
        self.add_tags_to_model(model_id, new_tags)

        return entities


mongo_handler = NLPService()

@app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.json
        article_id = data.get('article_id')
        text = data.get('text')
        model_id = data.get('model_id')
        print(article_id)
        print(model_id)
        print(text)

        if not article_id or not text or not model_id:
            return jsonify({'error': 'article_id, text, and model_id are required'}), 400

        # Process the text and save the extracted entities to the database using the provided model
        entities = mongo_handler.process_text_and_save_entities(article_id, text, model_id)
        print(jsonify({'message': 'Entities processed successfully', 'entities': entities}))
        return jsonify({'message': 'Entities processed successfully', 'entities': entities}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask application on port 8081
if __name__ == '__main__':
    app.run(port=8081, debug=True)
