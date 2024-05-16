import {useState} from 'react'
import axios from 'axios'
const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');

    const sendLogin = () =>{
        const data = {
            username : username,
            password : password,
            name : name
        }
        console.log(data)
        axios.post('http://localhost:5000/login',
            data
            
        ).then(res => console.log(res))
    }

    return (
        <div>
        blabla
        <form>
            <label for='username'>Username</label> 
        <input type="text" className="username" id="username" value={username} onChange={(e)=>setUsername(e.target.value)}></input>
        <label for='password'>password</label> 
        <input type="text" className="password" id="password" value={password} onChange={(e)=>setPassword(e.target.value)}></input>
        <label for='name'>name</label> 
        <input type="text" className="name" id="name" value={name} onChange={(e)=>setName(e.target.value)}></input>
        </form>
        <button onClick={()=>sendLogin()}>Send</button>
        </div>
    )
}
export default Login;