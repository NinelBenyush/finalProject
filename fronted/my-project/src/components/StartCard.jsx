import React from 'react';

function StartCard({ title, icon, text }) {
  return (
    <div className="bg-white p-8 rounded-md shadow-md">
      <div className="flex items-center gap-4 mb-4">
        {icon}
        <h3 className="text-xl font-semibold">{title}</h3>
      </div>
      <p>{text}</p>
    </div>
  );
}

export default StartCard;
