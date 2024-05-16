
import { nanoid } from 'nanoid';
import { LuFileSpreadsheet } from "react-icons/lu";
import { CiLogin } from "react-icons/ci";
import { IoIosNotifications } from "react-icons/io";


export const links = [
  { id: nanoid(), href: '#home', text: 'Home' },
  { id: nanoid(), href: '#start', text: 'Start' },
  { id: nanoid(), href: '#about', text: 'About' },
  { id: nanoid(), href: '#login', text: 'Sign up/Log in' },
];

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
    text: 'Second, you need to input your inventory data.',
  },
  {
    id: nanoid(),
    title: 'Notifications and Payment',
    icon: <IoIosNotifications className='h-16 w-16 text-emerald-500' />,
    text: 'Wait for updates and proceed with the payment.',
  },
];

export const projects = [
  {
    id: nanoid(),
    img: 'https://images.pexels.com/photos/326503/pexels-photo-326503.jpeg?auto=compress&cs=tinysrgb&w=800',
    url: 'https://react-projects.netlify.app/',
    github: 'https://github.com/john-smilga',
    title: 'first project',
    text: 'Lorem ipsum dolor sit amet consectetur, adipisicing elit. Asperiores aperiam porro impedit tenetur quo hic omnis doloribus dolores enim deleniti.',
  },
  {
    id: nanoid(),
    img: 'https://images.pexels.com/photos/2148222/pexels-photo-2148222.jpeg?auto=compress&cs=tinysrgb&w=800',
    url: 'https://react-projects.netlify.app/',
    github: 'https://github.com/john-smilga',
    title: 'second project',
    text: 'Lorem ipsum dolor sit amet consectetur, adipisicing elit. Asperiores aperiam porro impedit tenetur quo hic omnis doloribus dolores enim deleniti.',
  },
  {
    id: nanoid(),
    img: 'https://images.pexels.com/photos/12883026/pexels-photo-12883026.jpeg?auto=compress&cs=tinysrgb&w=800',
    url: 'https://react-projects.netlify.app/',
    github: 'https://github.com/john-smilga',
    title: 'third project',
    text: 'Lorem ipsum dolor sit amet consectetur, adipisicing elit. Asperiores aperiam porro impedit tenetur quo hic omnis doloribus dolores enim deleniti.',
  },
];
