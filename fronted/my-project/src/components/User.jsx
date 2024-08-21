import React from 'react';
//the user icon and name
const User = () => {
  const getUsername = () => {
    return localStorage.getItem('username') || 'A'; 
  };

  const username = localStorage.getItem('username');

  const getChar = () => {
    return username.charAt(0).toUpperCase();
  };

  return (
    <div className="avatar placeholder">
      <div className="bg-neutral text-neutral-content rounded-full w-8">
        <span className="text-xs">{getChar()}</span>
      </div>
    </div>
  );
};

export default User;
