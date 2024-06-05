import React, { useEffect, useState } from 'react';
import axios from 'axios';
import ProfileNavbar from './ProfileNavbar';
import Footer from './Footer';
import { FiMessageSquare } from "react-icons/fi";

const Updates = () => {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const savedMessages = JSON.parse(localStorage.getItem('messages'));
    if(savedMessages){
        setMessages(savedMessages);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('messages', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await axios.get('http://localhost:5000/profile/updates');
        setMessages(response.data);
      } catch (error) {
        console.error('Error fetching messages:', error);
      }
    };
  
    fetchMessages();
  }, []);

  console.log('State of messages:', messages);
  

  return (
    <>
      <ProfileNavbar />
      <div className="messages-container h-screen  p-4 bg-gray-100 rounded-md shadow-lg">
      <h2 className="text-2xl font-semibold mb-4 flex items-center">
       Messages
        <span className="ml-2"><FiMessageSquare className="h-7 w-7" /></span>
       </h2>

        {Array.isArray(messages) && messages.length > 0 ? (
            <ul className="space-y-2">
  {messages.map((message, index) => (
    <li key={index} className="p-2 bg-white rounded-md shadow-sm">
      <p className="text-gray-800">{message.content}</p>
      <span className="text-sm text-gray-500">{message.timestamp}</span>
    </li>
  ))}
</ul>

        ) : (
          <p className="text-gray-500">No messages yet</p>
        )}
      </div>
      <Footer />
 
    </>
  );
};

export default Updates;
