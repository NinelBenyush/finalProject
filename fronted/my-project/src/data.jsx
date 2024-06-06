
import { nanoid } from 'nanoid';
import { LuFileSpreadsheet } from "react-icons/lu";
import { CiLogin } from "react-icons/ci";
import { IoIosNotifications } from "react-icons/io";
import "./index.css";


export const links = [
  { id: nanoid(), href: '/', text: 'Home' },
  { id: nanoid(), href: '/register', text: 'Register' },
 // { id: nanoid(), href: '/login', text: 'Log in' },
];

export const profile = [
  {id:nanoid(), href:"/profile", text:'Profile'},
]

export const dropdownLinks = [
  { id: nanoid(), href: '#start', text: 'Start' },
  { id: nanoid(), href: '#about', text: 'About' },
];

export const profileLinks = [
  { id:nanoid(), href:"/basic-info", text:"Basic Information"},
  { id:nanoid(), href:"profile/messages", text:"Messages"},
];

export const handleFile = [
  {id:nanoid(), href:"/upload-file", text:"upload file"},
  {id:nanoid(), href:"/download-file", text:"download file"},
];

export const account = [
  {id:nanoid(), href:"/profile",text:"account"},
]

export const start = [
  {
    id: nanoid(),
    title: 'Log in/Sign Up',
    icon: <CiLogin className='h-16 w-16 text-emerald-500' />,
    text: 'First, you need to log in or sign in.',
  },
  {
    id: nanoid(),
    title: 'Insert Data',
    icon: <LuFileSpreadsheet className='h-16 w-16 text-emerald-500' />,
    text: ( 
      <>
      Second, you need to input your inventory data
    </>
    ),
  
  
  },
  {
    id: nanoid(),
    title: 'Notifications and Payment',
    icon: <IoIosNotifications className='h-16 w-16 text-emerald-500' />,
    text: 'Wait for updates and proceed with the payment.',
  },
];


