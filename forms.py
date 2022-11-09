from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from flask import Blueprint,flash,request,redirect,url_for,render_template
from whitenoise import WhiteNoise


forms= Blueprint("forms",__name__,static_folder="static", template_folder="templates")

class User:
    def __init__(self,user_id,email,password):
        self.user_id=user_id
        self.email=email
        self.password=password





class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

users=[]
users.append(User(user_id=1,email="admin@neptunesoftwaregroup.com",password="admin123"))

@forms.route("/", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = [x for x in users if x.email == form.email.data][0]

        if user and form.password.data == user.password:
            # flash('You have been logged in!', 'success')
            return redirect(url_for('models.create_task'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('application/test.html', title='Login', form=form)
