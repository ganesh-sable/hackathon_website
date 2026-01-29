# main.py - Complete AI-Powered Team Matching Platform (OpenRouter Version)
import streamlit as st
import sqlite3
import os
import json
import requests
import re
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import smtplib
from email.message import EmailMessage
from passlib.hash import pbkdf2_sha256
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
def init_session_state():
    defaults = {
        'user': None,
        'email': None,
        'profile_complete': False,
        'ai_generated': None,
        'show_registration': False,
        'current_project_id': None,
        'show_test_for': None,
        'page': 'login',
        'test_ai_result': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Configuration
DATABASE_NAME = "team_match.db"

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    # Try to get from Streamlit secrets
    try:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    except:
        st.error("""
        ❌ OPENROUTER_API_KEY not found. 
        
        Please create a `.env` file with:
        ```
        OPENROUTER_API_KEY=your_openrouter_api_key_here
        ```
        
        Or set Streamlit secrets.
        """)
        st.stop()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models on OpenRouter
OPENROUTER_MODELS = {
    "deepseek/deepseek-chat": "DeepSeek Chat (Recommended)",
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
    "openai/gpt-4o-mini": "GPT-4o Mini",
    "anthropic/claude-3-haiku": "Claude 3 Haiku",
    "anthropic/claude-3-sonnet": "Claude 3 Sonnet",
    "meta-llama/llama-3.1-8b-instruct": "LLaMA 3.1 8B",
    "mistralai/mistral-7b-instruct": "Mistral 7B"
}

# Default model - CORRECTED
DEFAULT_MODEL = "deepseek/deepseek-chat"

# Email Configuration (Optional)
EMAIL_CONFIG = {
    'user': os.getenv("EMAIL_USER"),
    'password': os.getenv("EMAIL_PASSWORD"),
    'server': os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    'port': int(os.getenv("SMTP_PORT", 587))
}

# Initialize Database
def init_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            email TEXT PRIMARY KEY,
            college TEXT NOT NULL,
            class_year TEXT NOT NULL,
            age INTEGER NOT NULL,
            location TEXT NOT NULL,
            skills TEXT NOT NULL,
            experience TEXT NOT NULL,
            bio TEXT,
            github_url TEXT,
            linkedin_url TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    ''')
    
    # Projects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY AUTOINCREMENT,
            creator_email TEXT NOT NULL,
            project_name TEXT NOT NULL,
            project_description TEXT,
            required_roles TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            team_size INTEGER NOT NULL,
            experience_level TEXT NOT NULL,
            duration TEXT NOT NULL,
            filter_option TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (creator_email) REFERENCES users(email)
        )
    ''')
    
    # Project invites table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_invites (
            invite_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (user_email) REFERENCES users(email),
            UNIQUE(project_id, user_email)
        )
    ''')
    
    # Test responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            test_questions TEXT,
            user_answers TEXT,
            ai_score REAL,
            feedback TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (user_email) REFERENCES users(email)
        )
    ''')
    
    # Project members table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_members (
            member_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            role TEXT,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (user_email) REFERENCES users(email),
            UNIQUE(project_id, user_email)
        )
    ''')
    
    # Add settings table for AI model preferences
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            email TEXT PRIMARY KEY,
            preferred_ai_model TEXT DEFAULT 'deepseek/deepseek-chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# Database Manager
class Database:
    @staticmethod
    def execute(query, params=(), fetchone=False, fetchall=False):
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()
        else:
            result = None
        
        conn.commit()
        conn.close()
        return result
    
    # User methods
    @classmethod
    def create_user(cls, email, password, name):
        if cls.get_user(email):
            return False
        password_hash = pbkdf2_sha256.hash(password)
        result = cls.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email, password_hash, name)
        )
        if result is not None:
            # Initialize user settings
            cls.execute(
                "INSERT OR IGNORE INTO user_settings (email, preferred_ai_model) VALUES (?, ?)",
                (email, DEFAULT_MODEL)
            )
        return result
    
    @classmethod
    def get_user(cls, email):
        return cls.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,), fetchone=True
        )
    
    @classmethod
    def verify_user(cls, email, password):
        user = cls.get_user(email)
        if user:
            return pbkdf2_sha256.verify(password, user[1])
        return False
    
    # Profile methods
    @classmethod
    def save_profile(cls, email, **data):
        exists = cls.execute(
            "SELECT email FROM profiles WHERE email = ?",
            (email,), fetchone=True
        )
        
        fields = ['college', 'class_year', 'age', 'location', 'skills', 
                 'experience', 'bio', 'github_url', 'linkedin_url']
        values = [data.get(field, '') for field in fields]
        
        if exists:
            query = f'''UPDATE profiles SET {
                ', '.join([f'{field}=?' for field in fields])
            }, updated_at=CURRENT_TIMESTAMP WHERE email=?'''
            return cls.execute(query, values + [email])
        else:
            query = f'''INSERT INTO profiles (email, {
                ', '.join(fields)
            }) VALUES ({', '.join(['?']*(len(fields)+1))})'''
            return cls.execute(query, [email] + values)
    
    @classmethod
    def get_profile(cls, email):
        return cls.execute(
            "SELECT * FROM profiles WHERE email = ?",
            (email,), fetchone=True
        )
    
    # Settings methods
    @classmethod
    def get_user_settings(cls, email):
        return cls.execute(
            "SELECT * FROM user_settings WHERE email = ?",
            (email,), fetchone=True
        )
    
    @classmethod
    def update_ai_model(cls, email, model_name):
        return cls.execute(
            "UPDATE user_settings SET preferred_ai_model = ? WHERE email = ?",
            (model_name, email)
        )
    
    # Project methods
    @classmethod
    def create_project(cls, creator_email, **data):
        fields = ['project_name', 'project_description', 'required_roles',
                 'required_skills', 'team_size', 'experience_level',
                 'duration', 'filter_option']
        
        query = f'''INSERT INTO projects (creator_email, {
            ', '.join(fields)
        }) VALUES ({', '.join(['?']*(len(fields)+1))})'''
        
        values = [creator_email] + [data.get(field, '') for field in fields]
        cls.execute(query, values)
        
        result = cls.execute("SELECT last_insert_rowid()", fetchone=True)
        return result[0] if result else None
    
    @classmethod
    def get_user_projects(cls, email):
        return cls.execute(
            "SELECT * FROM projects WHERE creator_email = ? ORDER BY created_at DESC",
            (email,), fetchall=True
        )
    
    @classmethod
    def get_project(cls, project_id):
        return cls.execute(
            "SELECT * FROM projects WHERE project_id = ?",
            (project_id,), fetchone=True
        )
    
    @classmethod
    def get_all_projects(cls, exclude_email=None):
        if exclude_email:
            return cls.execute(
                "SELECT * FROM projects WHERE creator_email != ? AND status = 'active' ORDER BY created_at DESC",
                (exclude_email,), fetchall=True
            )
        return cls.execute(
            "SELECT * FROM projects WHERE status = 'active' ORDER BY created_at DESC",
            fetchall=True
        )
    
    # Invite methods
    @classmethod
    def save_invite(cls, project_id, user_email, message=""):
        try:
            return cls.execute(
                """INSERT OR IGNORE INTO project_invites 
                   (project_id, user_email, message) VALUES (?, ?, ?)""",
                (project_id, user_email, message)
            ) is not None
        except Exception as e:
            st.error(f"Invite error: {str(e)}")
            return False
    
    @classmethod
    def update_invite_status(cls, invite_id, status):
        return cls.execute(
            "UPDATE project_invites SET status = ? WHERE invite_id = ?",
            (status, invite_id)
        )
    
    @classmethod
    def get_pending_invites(cls, email):
        return cls.execute(
            """SELECT pi.*, p.project_name, p.creator_email, u.name as creator_name
               FROM project_invites pi
               JOIN projects p ON pi.project_id = p.project_id
               JOIN users u ON p.creator_email = u.email
               WHERE pi.user_email = ? AND pi.status = 'pending'
               ORDER BY pi.invited_at DESC""",
            (email,), fetchall=True
        )
    
    # Search methods
    @classmethod
    def search_profiles(cls, exclude_email=None, filters=None):
        query = "SELECT * FROM profiles WHERE 1=1"
        params = []
        
        if exclude_email:
            query += " AND email != ?"
            params.append(exclude_email)
        
        if filters:
            if filters.get('skills'):
                skills = [s.strip() for s in filters['skills'].split(',')]
                query += " AND ("
                for i, skill in enumerate(skills):
                    if i > 0:
                        query += " OR "
                    query += "skills LIKE ?"
                    params.append(f"%{skill}%")
                query += ")"
            
            if filters.get('college'):
                query += " AND college LIKE ?"
                params.append(f"%{filters['college']}%")
            
            if filters.get('location'):
                query += " AND location LIKE ?"
                params.append(f"%{filters['location']}%")
        
        return cls.execute(query, params, fetchall=True)

# OpenRouter AI Integration
class OpenRouterAI:
    @staticmethod
    def get_user_model(email):
        """Get user's preferred AI model"""
        settings = Database.get_user_settings(email)
        if settings and settings[1]:
            return settings[1]
        return DEFAULT_MODEL
    
    @staticmethod
    def generate_project_details(project_name, model_name=None, email=None):
        """Generate project details using OpenRouter API"""
        
        if email and not model_name:
            model_name = OpenRouterAI.get_user_model(email)
        elif not model_name:
            model_name = DEFAULT_MODEL
        
        # First detect project type
        project_type = OpenRouterAI.detect_project_type(project_name)
        
        # Create prompt based on project type
        prompt = f"""Generate team composition for project: "{project_name}"

This is a {project_type} project. Please provide:
1. Required team roles (comma-separated)
2. Required technical skills (comma-separated)
3. Recommended team size (single number 1-10)
4. Recommended experience level (Beginner, Intermediate, or Advanced)
5. Estimated duration (e.g., "1-2 months", "3-6 months")

Format exactly:
Roles: [roles here]
Skills: [skills here]
Team Size: [number]
Experience Level: [level]
Duration: [duration]"""
        
        # Call OpenRouter API
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Team Match AI"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a project planning expert. Generate appropriate team composition for tech projects. Respond only with the requested format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            parsed_result = OpenRouterAI.parse_response(content)
            
            # Store model used
            parsed_result['model_used'] = model_name
            return parsed_result
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: {str(e)[:100]}")
            fallback = OpenRouterAI.get_fallback_template(project_type)
            fallback['model_used'] = "fallback"
            return fallback
        except Exception as e:
            st.error(f"AI Generation Error: {str(e)[:100]}")
            fallback = OpenRouterAI.get_fallback_template(project_type)
            fallback['model_used'] = "fallback"
            return fallback
    
    @staticmethod
    def detect_project_type(project_name):
        """Detect project type from name"""
        name_lower = project_name.lower()
        
        project_types = [
            (["medical", "health", "patient", "hospital"], "medical/healthcare"),
            (["ml", "machine learning", "data science", "ai", "analytics"], "data science/machine learning"),
            (["web", "website", "ecommerce", "portal", "shop"], "web development"),
            (["mobile", "android", "ios", "app", "flutter"], "mobile development"),
            (["weather", "climate", "forecast"], "weather/forecast"),
            (["game", "gaming", "unity", "unreal"], "game development"),
            (["iot", "internet of things", "smart", "device", "arduino", "raspberry"], "IoT/embedded systems"),
            (["chatbot", "bot", "assistant", "virtual"], "chatbot/assistant"),
            (["blockchain", "crypto", "web3", "nft"], "blockchain"),
            (["education", "learning", "course", "tutorial"], "education tech")
        ]
        
        for keywords, p_type in project_types:
            if any(keyword in name_lower for keyword in keywords):
                return p_type
        
        return "general"
    
    @staticmethod
    def parse_response(content):
        """Parse AI response"""
        result = {
            "roles": "Project Manager, Developer, Designer",
            "skills": "Python, JavaScript, UI/UX",
            "team_size": 3,
            "experience_level": "Intermediate",
            "duration": "2-3 months",
            "model_used": "unknown"
        }
        
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Roles:'):
                result['roles'] = line.replace('Roles:', '').strip()
            elif line.startswith('Skills:'):
                result['skills'] = line.replace('Skills:', '').strip()
            elif line.startswith('Team Size:'):
                try:
                    result['team_size'] = int(''.join(filter(str.isdigit, line.replace('Team Size:', ''))))
                except:
                    result['team_size'] = 3
            elif line.startswith('Experience Level:'):
                level = line.replace('Experience Level:', '').strip()
                if level in ["Beginner", "Intermediate", "Advanced"]:
                    result['experience_level'] = level
            elif line.startswith('Duration:'):
                result['duration'] = line.replace('Duration:', '').strip()
        
        return result
    
    @staticmethod
    def get_fallback_template(project_type):
        """Get fallback template based on project type"""
        templates = {
            "medical/healthcare": {
                "roles": "Medical NLP Engineer, Healthcare Developer, UI/UX Designer, Domain Expert",
                "skills": "Python, NLP, Healthcare APIs, Security, React, PostgreSQL",
                "team_size": 4,
                "experience_level": "Intermediate",
                "duration": "4-6 months"
            },
            "data science/machine learning": {
                "roles": "Data Scientist, ML Engineer, Data Analyst, Python Developer",
                "skills": "Python, Pandas, NumPy, Scikit-learn, Matplotlib, Statistics",
                "team_size": 3,
                "experience_level": "Intermediate",
                "duration": "3-4 months"
            },
            "weather/forecast": {
                "roles": "Data Scientist, Python Developer, Backend Developer, Data Analyst",
                "skills": "Python, Pandas, Matplotlib, Machine Learning, Statistics, API Integration",
                "team_size": 4,
                "experience_level": "Intermediate",
                "duration": "3-4 months"
            },
            "web development": {
                "roles": "Frontend Developer, Backend Developer, UI/UX Designer",
                "skills": "JavaScript, React, Node.js, MongoDB, HTML/CSS, Figma",
                "team_size": 3,
                "experience_level": "Intermediate",
                "duration": "2-3 months"
            },
            "mobile development": {
                "roles": "Android Developer, iOS Developer, Backend Developer, UI/UX Designer",
                "skills": "Kotlin, Swift, Java, React Native, Firebase, REST APIs",
                "team_size": 3,
                "experience_level": "Intermediate",
                "duration": "3-4 months"
            },
            "game development": {
                "roles": "Game Developer, 3D Artist, Sound Designer, UI/UX Designer",
                "skills": "Unity, C#, Blender, Photoshop, Game Physics",
                "team_size": 4,
                "experience_level": "Intermediate",
                "duration": "4-6 months"
            },
            "blockchain": {
                "roles": "Blockchain Developer, Smart Contract Developer, Backend Developer, Security Auditor",
                "skills": "Solidity, Web3.js, Ethereum, Node.js, Cryptography",
                "team_size": 3,
                "experience_level": "Advanced",
                "duration": "4-6 months"
            }
        }
        
        return templates.get(project_type, {
            "roles": "Project Manager, Developer, Designer, Tester",
            "skills": "Python, JavaScript, Database, Problem Solving",
            "team_size": 3,
            "experience_level": "Intermediate",
            "duration": "2-3 months"
        })
    
    @staticmethod
    def calculate_skill_match(user_skills, required_skills):
        """Calculate skill match percentage"""
        if not user_skills or not required_skills:
            return 0
        
        user_skills_list = [s.strip().lower() for s in str(user_skills).split(',')]
        required_skills_list = [s.strip().lower() for s in str(required_skills).split(',')]
        
        if not required_skills_list:
            return 0
        
        matches = 0
        for req_skill in required_skills_list:
            for user_skill in user_skills_list:
                if req_skill and user_skill:
                    if (req_skill in user_skill or user_skill in req_skill):
                        matches += 1
                        break
        
        return round((matches / len(required_skills_list)) * 100, 1)

# UI Components
def show_login():
    """Login page"""
    st.title("🔐 Login to Team Match")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            col_btn = st.columns(2)
            with col_btn[0]:
                login = st.form_submit_button("Login", use_container_width=True)
            with col_btn[1]:
                register = st.form_submit_button("Register", use_container_width=True)
            
            if login:
                if email and password:
                    if Database.verify_user(email, password):
                        st.session_state.user = email
                        st.session_state.email = email
                        st.session_state.page = "dashboard"
                        profile = Database.get_profile(email)
                        st.session_state.profile_complete = profile is not None
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.warning("Please fill all fields")
            
            if register:
                st.session_state.page = "register"
                st.rerun()
    
    with col2:
        st.info("""
        **Welcome!** 🤝
        
        AI-powered team matching platform.
        
        Features:
        • AI-generated project roles
        • Smart skill matching
        • Team collaboration
        • Project management
        • Multiple AI Models
        """)

def show_register():
    """Registration page"""
    st.title("📝 Create Account")
    
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        
        col_btn = st.columns(2)
        with col_btn[0]:
            submit = st.form_submit_button("Register", use_container_width=True)
        with col_btn[1]:
            back = st.form_submit_button("Back to Login", use_container_width=True)
        
        if submit:
            if not all([name, email, password, confirm]):
                st.warning("Please fill all fields")
            elif password != confirm:
                st.error("Passwords don't match")
            elif Database.get_user(email):
                st.error("Email already registered")
            else:
                if Database.create_user(email, password, name):
                    st.success("Account created! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
        
        if back:
            st.session_state.page = "login"
            st.rerun()

def show_profile():
    """Profile management"""
    st.title("👤 Your Profile")
    
    profile = Database.get_profile(st.session_state.user)
    user = Database.get_user(st.session_state.user)
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name", value=user[2] if user else "")
            college = st.text_input("College", value=profile[1] if profile else "")
            class_year = st.selectbox(
                "Class Year",
                ["Freshman", "Sophomore", "Junior", "Senior", "Graduate", "Other"],
                index=0 if not profile else 0
            )
            age = st.number_input("Age", min_value=16, max_value=80,
                                 value=profile[3] if profile else 20)
            location = st.text_input("Location", value=profile[4] if profile else "")
        
        with col2:
            skills = st.text_area("Skills (comma-separated)",
                                 value=profile[5] if profile else "",
                                 placeholder="Python, JavaScript, React, UI/UX...",
                                 height=100)
            experience = st.selectbox(
                "Experience Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=0 if not profile else 0
            )
            bio = st.text_area("Bio", value=profile[7] if profile else "",
                              height=80)
            github = st.text_input("GitHub", value=profile[8] if profile else "")
            linkedin = st.text_input("LinkedIn", value=profile[9] if profile else "")
        
        save = st.form_submit_button("Save Profile", use_container_width=True)
        
        if save:
            if all([name, college, location, skills]):
                Database.save_profile(
                    st.session_state.user,
                    college=college,
                    class_year=class_year,
                    age=age,
                    location=location,
                    skills=skills,
                    experience=experience,
                    bio=bio,
                    github_url=github,
                    linkedin_url=linkedin
                )
                st.session_state.profile_complete = True
                st.success("Profile saved!")
                st.rerun()
            else:
                st.error("Please fill required fields")

def show_ai_settings():
    """AI Model Settings"""
    st.title("⚙️ AI Settings")
    
    settings = Database.get_user_settings(st.session_state.user)
    current_model = settings[1] if settings else DEFAULT_MODEL
    
    st.markdown("### Choose Your Preferred AI Model")
    st.info("Different models may have different capabilities and response times.")
    
    selected_model = st.selectbox(
        "AI Model",
        options=list(OPENROUTER_MODELS.keys()),
        format_func=lambda x: OPENROUTER_MODELS[x],
        index=list(OPENROUTER_MODELS.keys()).index(current_model) if current_model in OPENROUTER_MODELS else 0
    )
    
    # Show model information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Model", OPENROUTER_MODELS.get(current_model, current_model))
    with col2:
        st.metric("Selected Model", OPENROUTER_MODELS.get(selected_model, selected_model))
    
    # Save button
    if st.button("💾 Save Model Preference", use_container_width=True):
        Database.update_ai_model(st.session_state.user, selected_model)
        st.success(f"✅ Model preference updated to {OPENROUTER_MODELS.get(selected_model, selected_model)}")
        st.rerun()
    
    # Test the model
    st.divider()
    st.markdown("### 🤖 Test AI Model")
    
    test_input = st.text_input("Enter a project name to test:", placeholder="e.g., E-commerce Website, Medical AI Assistant")
    
    col_test, col_clear = st.columns(2)
    with col_test:
        if st.button("🚀 Test Model", key="test_ai_model", use_container_width=True):
            if test_input:
                with st.spinner(f"Testing {OPENROUTER_MODELS.get(selected_model, selected_model)}..."):
                    result = OpenRouterAI.generate_project_details(
                        test_input, 
                        model_name=selected_model,
                        email=st.session_state.user
                    )
                    
                    st.session_state.test_ai_result = result
            else:
                st.warning("Please enter a project name to test")
    
    with col_clear:
        if st.button("🔄 Clear Test", use_container_width=True):
            st.session_state.test_ai_result = None
            st.rerun()
    
    # Display test result
    if st.session_state.test_ai_result:
        st.success("✅ AI Response Generated Successfully!")
        
        result = st.session_state.test_ai_result
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### 📋 AI Suggestions")
            st.write(f"**Model Used:** {result.get('model_used', 'Unknown')}")
            st.write(f"**Roles:** {result['roles']}")
            st.write(f"**Skills:** {result['skills']}")
        
        with col_res2:
            st.markdown("#### 📊 Details")
            st.write(f"**Team Size:** {result['team_size']}")
            st.write(f"**Experience Level:** {result['experience_level']}")
            st.write(f"**Duration:** {result['duration']}")

def show_create_project():
    """Create new project with AI"""
    st.title("🚀 Create New Project")
    
    # Get user's preferred AI model
    settings = Database.get_user_settings(st.session_state.user)
    ai_model = settings[1] if settings else DEFAULT_MODEL
    
    st.info(f"🤖 Using AI Model: **{OPENROUTER_MODELS.get(ai_model, ai_model)}**")
    
    # Project name input
    project_name = st.text_input(
        "Project Name",
        placeholder="Enter your project name (e.g., Medical Report Simplifier, Weather Forecast ML)"
    )
    
    col_gen, col_manual = st.columns(2)
    
    with col_gen:
        # AI Generate button
        if st.button("🤖 Generate with AI", key="ai_generate", use_container_width=True):
            if project_name:
                with st.spinner(f"{OPENROUTER_MODELS.get(ai_model, ai_model)} is generating project details..."):
                    ai_details = OpenRouterAI.generate_project_details(
                        project_name, 
                        model_name=ai_model,
                        email=st.session_state.user
                    )
                    st.session_state.ai_generated = ai_details
                    st.rerun()
            else:
                st.warning("Please enter project name first")
    
    with col_manual:
        # Clear AI suggestions
        if st.session_state.ai_generated:
            if st.button("🔄 Clear AI Suggestions", key="clear_ai", use_container_width=True):
                st.session_state.ai_generated = None
                st.rerun()
    
    # Main form
    with st.form("project_form"):
        st.subheader("Project Details")
        
        # Pre-fill with AI data if available
        if st.session_state.ai_generated:
            ai_data = st.session_state.ai_generated
            st.success("✅ AI Suggestions Loaded!")
            
            roles = st.text_area(
                "Required Roles (AI Suggested)",
                value=ai_data['roles'],
                height=80
            )
            skills = st.text_area(
                "Required Skills (AI Suggested)",
                value=ai_data['skills'],
                height=100
            )
            
            col_size, col_exp, col_dur = st.columns(3)
            with col_size:
                team_size = st.number_input(
                    "Team Size",
                    min_value=1, max_value=20,
                    value=ai_data['team_size']
                )
            with col_exp:
                experience = st.selectbox(
                    "Experience Level",
                    ["Beginner", "Intermediate", "Advanced"],
                    index=["Beginner", "Intermediate", "Advanced"].index(
                        ai_data['experience_level']
                    ) if ai_data['experience_level'] in 
                    ["Beginner", "Intermediate", "Advanced"] else 1
                )
            with col_dur:
                duration = st.text_input(
                    "Duration",
                    value=ai_data['duration']
                )
        else:
            roles = st.text_area(
                "Required Roles",
                placeholder="e.g., Frontend Developer, Backend Developer, UI/UX Designer",
                height=80
            )
            skills = st.text_area(
                "Required Skills",
                placeholder="e.g., React, Python, MongoDB, Figma",
                height=100
            )
            team_size = st.number_input("Team Size", min_value=1, max_value=20, value=3)
            experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
            duration = st.text_input("Duration", placeholder="e.g., 2-3 months")
        
        description = st.text_area("Project Description", height=100, 
                                  placeholder="Describe your project goals, objectives, and vision...")
        
        filter_option = st.radio(
            "Search Team Members From",
            ["My College Only", "All Users", "Location Based"]
        )
        
        col_submit, col_cancel = st.columns(2)
        with col_submit:
            create = st.form_submit_button("🚀 Create Project", use_container_width=True)
        with col_cancel:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
        
        if create:
            if all([project_name, roles, skills, duration]):
                project_id = Database.create_project(
                    st.session_state.user,
                    project_name=project_name,
                    project_description=description,
                    required_roles=roles,
                    required_skills=skills,
                    team_size=team_size,
                    experience_level=experience,
                    duration=duration,
                    filter_option=filter_option
                )
                
                if project_id:
                    st.success(f"✅ Project created! ID: {project_id}")
                    
                    if st.button("🔍 Find Team Members Now", key="find_after_create"):
                        st.session_state.current_project_id = project_id
                        st.session_state.page = "find_candidates"
                        st.rerun()
            else:
                st.error("Please fill all required fields")

def show_my_projects():
    """Show user's projects"""
    st.title("📋 My Projects")
    
    projects = Database.get_user_projects(st.session_state.user)
    
    if not projects:
        st.info("You haven't created any projects yet.")
        if st.button("🚀 Create Your First Project"):
            st.session_state.page = "create_project"
            st.rerun()
        return
    
    for project in projects:
        with st.expander(f"{project[2]} (ID: {project[0]}) - {project[10]}", expanded=True):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.write(f"**Description:** {project[3] or 'No description'}")
                st.write(f"**Team Size:** {project[5]} members")
                st.write(f"**Skills Needed:** {project[4]}")
                st.write(f"**Status:** {project[10]}")
                st.write(f"**Created:** {project[11]}")
            
            with col_actions:
                if st.button("🔍 Find Candidates", key=f"find_{project[0]}"):
                    st.session_state.current_project_id = project[0]
                    st.session_state.page = "find_candidates"
                    st.rerun()
                
                if st.button("📊 View Details", key=f"view_{project[0]}"):
                    st.session_state.current_project_id = project[0]
                    st.session_state.page = "project_details"
                    st.rerun()

def show_find_candidates():
    """Find candidates for project"""
    if not st.session_state.current_project_id:
        show_my_projects()
        return
    
    project = Database.get_project(st.session_state.current_project_id)
    
    if not project:
        st.error("Project not found")
        st.session_state.current_project_id = None
        st.rerun()
        return
    
    st.title(f"👥 Find Team for: {project[2]}")
    
    with st.container(border=True):
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Required Skills", project[4].count(',') + 1)
        with col_sum2:
            st.metric("Team Size", project[5])
        with col_sum3:
            st.metric("Duration", project[8])
    
    # Search filters
    with st.expander("🔍 Advanced Search Filters"):
        search_skills = st.text_input("Filter by Skills", value=project[4])
        search_college = st.text_input("Filter by College")
        search_location = st.text_input("Filter by Location")
        min_match = st.slider("Minimum Match %", 0, 100, 50)
    
    # Search button
    if st.button("🔍 Search Candidates", use_container_width=True, type="primary"):
        filters = {}
        if search_skills:
            filters['skills'] = search_skills
        if search_college:
            filters['college'] = search_college
        if search_location:
            filters['location'] = search_location
        
        profiles = Database.search_profiles(
            exclude_email=st.session_state.user,
            filters=filters
        )
        
        if not profiles:
            st.info("No candidates found matching your criteria.")
            return
        
        st.subheader(f"🎯 Found {len(profiles)} Candidates")
        
        # Display candidates
        displayed = 0
        for profile in profiles:
            email, college, class_year, age, location, skills, exp, bio, github, linkedin, updated_at = profile
            
            # Calculate match score
            match_score = OpenRouterAI.calculate_skill_match(skills, project[4])
            
            if match_score >= min_match:
                displayed += 1
                with st.container(border=True):
                    col_info, col_match = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**📧 {email}**")
                        st.write(f"🎓 **College:** {college} ({class_year})")
                        st.write(f"📍 **Location:** {location}")
                        st.write(f"🎯 **Experience:** {exp}")
                        st.write(f"🛠️ **Skills:** {skills[:150]}...")
                        if bio:
                            st.write(f"📝 **Bio:** {bio[:100]}...")
                    
                    with col_match:
                        # Color code based on match score
                        if match_score >= 80:
                            color = "🟢"
                        elif match_score >= 60:
                            color = "🟡"
                        else:
                            color = "🟠"
                        
                        st.markdown(f"### {color} {match_score}%")
                        st.progress(match_score / 100)
                        
                        # Action buttons
                        if st.button("📨 Invite", key=f"invite_{email}_{displayed}", use_container_width=True):
                            if Database.save_invite(
                                st.session_state.current_project_id,
                                email,
                                f"Invitation to join {project[2]} project"
                            ):
                                st.success(f"✅ Invitation sent to {email}")
                                st.rerun()
        
        if displayed == 0:
            st.warning(f"No candidates found with minimum {min_match}% match score.")
    
    # Back button
    if st.button("← Back to My Projects"):
        st.session_state.current_project_id = None
        st.session_state.page = "my_projects"
        st.rerun()

def show_invitations():
    """Show user's invitations"""
    st.title("📨 My Invitations")
    
    invites = Database.get_pending_invites(st.session_state.user)
    
    if not invites:
        st.info("🎉 No pending invitations. Keep building your profile to get more invites!")
        return
    
    st.info(f"📬 You have {len(invites)} pending invitation(s)")
    
    for invite in invites:
        invite_id, project_id, user_email, status, invited_at, message, project_name, creator_email, creator_name = invite
        
        with st.container(border=True):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"### 📋 {project_name}")
                st.write(f"**From:** 👤 {creator_name} ({creator_email})")
                st.write(f"**Invited on:** 📅 {invited_at}")
                if message:
                    st.write(f"**Message:** 💬 {message}")
                
                # Get project details
                project = Database.get_project(project_id)
                if project:
                    st.write(f"**Project Skills:** 🛠️ {project[4][:100]}...")
            
            with col_actions:
                col_acc, col_dec = st.columns(2)
                with col_acc:
                    if st.button("✅ Accept", key=f"acc_{invite_id}", use_container_width=True):
                        Database.update_invite_status(invite_id, 'accepted')
                        st.success("🎉 Invitation accepted! You're now part of the project.")
                        st.rerun()
                with col_dec:
                    if st.button("❌ Decline", key=f"dec_{invite_id}", use_container_width=True):
                        Database.update_invite_status(invite_id, 'rejected')
                        st.success("Invitation declined")
                        st.rerun()

def show_dashboard():
    """Main dashboard"""
    st.title("🏠 Dashboard")
    
    # User info
    user = Database.get_user(st.session_state.user)
    profile = Database.get_profile(st.session_state.user)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        projects = Database.get_user_projects(st.session_state.user)
        st.metric("My Projects", len(projects))
    with col2:
        invites = Database.get_pending_invites(st.session_state.user)
        st.metric("Pending Invites", len(invites))
    with col3:
        st.metric("Profile Status", "✅ Complete" if profile else "❌ Incomplete")
    with col4:
        settings = Database.get_user_settings(st.session_state.user)
        model_name = settings[1] if settings else DEFAULT_MODEL
        model_display = OPENROUTER_MODELS.get(model_name, model_name)
        st.metric("AI Model", model_display.split()[0])
    
    # Welcome message
    if user:
        st.subheader(f"👋 Welcome back, {user[2]}!")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col_actions = st.columns(4)
    actions = [
        ("🚀 Create Project", "create_project", "Start a new project"),
        ("🔍 Find Team", "find_candidates", "Find team members"),
        ("👤 Profile", "profile", "Update your profile"),
        ("⚙️ AI Settings", "ai_settings", "Configure AI model")
    ]
    
    for i, (text, page, tooltip) in enumerate(actions):
        with col_actions[i]:
            if st.button(text, use_container_width=True, help=tooltip):
                st.session_state.page = page
                st.rerun()
    
    # Recent projects
    st.markdown("### 📋 Recent Projects")
    projects = Database.get_user_projects(st.session_state.user)[:5]
    
    if projects:
        for project in projects:
            with st.container(border=True):
                col_proj, col_btn = st.columns([3, 1])
                
                with col_proj:
                    st.write(f"**{project[2]}**")
                    st.write(f"🛠️ {project[4][:80]}...")
                    st.caption(f"Created: {project[11]}")
                
                with col_btn:
                    if st.button("View", key=f"dash_{project[0]}"):
                        st.session_state.current_project_id = project[0]
                        st.session_state.page = "project_details"
                        st.rerun()
    else:
        st.info("You haven't created any projects yet. Click 'Create Project' to get started!")
    
    # Available projects from others
    st.markdown("### 🌟 Available Projects")
    other_projects = Database.get_all_projects(exclude_email=st.session_state.user)[:3]
    
    if other_projects:
        for project in other_projects:
            with st.container(border=True):
                col_proj, col_match = st.columns([3, 1])
                
                with col_proj:
                    st.write(f"**{project[2]}**")
                    st.write(f"👤 By: {project[1]}")
                    st.write(f"🛠️ {project[4][:60]}...")
                
                with col_match:
                    # Check if user has matching skills
                    profile = Database.get_profile(st.session_state.user)
                    if profile:
                        match_score = OpenRouterAI.calculate_skill_match(profile[5], project[4])
                        st.metric("Match", f"{match_score}%")
                    else:
                        st.write("Complete profile to see match")
    else:
        st.info("No other projects available at the moment.")

def show_project_details():
    """Project details page"""
    if not st.session_state.current_project_id:
        st.warning("No project selected")
        st.session_state.page = "my_projects"
        st.rerun()
        return
    
    project = Database.get_project(st.session_state.current_project_id)
    
    if not project:
        st.error("Project not found")
        st.session_state.current_project_id = None
        st.rerun()
        return
    
    st.title(f"📊 Project: {project[2]}")
    
    # Project details in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Project Details")
        st.info(f"**Project ID:** {project[0]}")
        st.info(f"**Created By:** {project[1]}")
        st.info(f"**Status:** {project[10]}")
        st.info(f"**Created:** {project[11]}")
        
        st.markdown("### 🎯 Requirements")
        st.info(f"**Experience Level:** {project[7]}")
        st.info(f"**Team Size:** {project[5]}")
        st.info(f"**Duration:** {project[8]}")
        st.info(f"**Search Filter:** {project[9]}")
    
    with col2:
        st.markdown("### 👥 Team Composition")
        st.success(f"**Required Roles:**\n{project[3]}")
        st.success(f"**Required Skills:**\n{project[4]}")
    
    st.markdown("### 📝 Project Description")
    if project[3]:
        st.write(project[3])
    else:
        st.write("*No description provided*")
    
    # Action buttons
    st.divider()
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("🔍 Find Candidates", use_container_width=True):
            st.session_state.page = "find_candidates"
            st.rerun()
    
    with col_btn2:
        if st.button("📨 View Invitations", use_container_width=True):
            st.session_state.page = "invitations"
            st.rerun()
    
    with col_btn3:
        if st.button("✏️ Edit Project", use_container_width=True):
            st.warning("Edit feature coming soon!")
    
    with col_btn4:
        if st.button("← Back to Projects", use_container_width=True):
            st.session_state.current_project_id = None
            st.session_state.page = "my_projects"
            st.rerun()

# Main App
def main():
    # Page configuration
    st.set_page_config(
        page_title="Team Match - AI Project Team Matching",
        page_icon="🤝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stButton > button {
        border-radius: 10px;
        height: 3em;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🤝 Team Match")
        
        if st.session_state.user:
            user = Database.get_user(st.session_state.user)
            if user:
                st.markdown(f"### 👤 {user[2]}")
                st.caption(f"📧 {user[0]}")
            
            # Show AI model info
            settings = Database.get_user_settings(st.session_state.user)
            if settings:
                model_name = settings[1]
                st.caption(f"🤖 AI: {OPENROUTER_MODELS.get(model_name, model_name)}")
            
            st.divider()
            
            # Navigation menu
            pages = {
                "🏠 Dashboard": "dashboard",
                "👤 Profile": "profile",
                "⚙️ AI Settings": "ai_settings",
                "🚀 Create Project": "create_project",
                "📋 My Projects": "my_projects",
                "🔍 Find Team": "find_candidates",
                "📨 Invitations": "invitations"
            }
            
            selected = st.radio("Navigation", list(pages.keys()))
            st.session_state.page = pages[selected]
            
            st.divider()
            
            # Quick stats
            if st.session_state.user:
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    projects = len(Database.get_user_projects(st.session_state.user))
                    st.metric("Projects", projects, label_visibility="collapsed")
                with col_stat2:
                    invites = len(Database.get_pending_invites(st.session_state.user))
                    st.metric("Invites", invites, label_visibility="collapsed")
            
            st.divider()
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session_state()
                st.rerun()
        else:
            st.info("Please login to continue")
            st.image("https://cdn-icons-png.flaticon.com/512/1006/1006771.png", width=100)
            st.caption("AI-Powered Team Matching Platform")
    
    # Main content
    if not st.session_state.user:
        if st.session_state.page == "register":
            show_register()
        else:
            show_login()
    else:
        # Check profile completion
        profile = Database.get_profile(st.session_state.user)
        if not profile and st.session_state.page not in ["profile", "ai_settings", "login", "register"]:
            st.warning("⚠️ Please complete your profile first!")
            st.session_state.page = "profile"
        
        # Page routing
        page_handlers = {
            "dashboard": show_dashboard,
            "profile": show_profile,
            "ai_settings": show_ai_settings,
            "create_project": show_create_project,
            "my_projects": show_my_projects,
            "find_candidates": show_find_candidates,
            "invitations": show_invitations,
            "project_details": show_project_details
        }
        
        handler = page_handlers.get(st.session_state.page, show_dashboard)
        handler()

if __name__ == "__main__":
    main()