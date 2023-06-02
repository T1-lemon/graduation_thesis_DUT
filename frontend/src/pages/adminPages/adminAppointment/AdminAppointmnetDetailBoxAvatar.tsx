import Avatar from '@mui/material/Avatar'
import React from 'react'

type Props = {
  name: string
  profilePicture: string
}

const AdminAppointmnetDetailBoxAvatar = (props: Props) => {
  const { name, profilePicture } = props
  return (
    <div className='admin__apptdetail--bodyleft'>
      <Avatar
        src={profilePicture}
        sx={{
          width: 90,
          height: 90,
          border: '1px solid var(--color-text)'
        }}
      />
      <p className='bodyleft__p'>{name}</p>
    </div>
  )
}

export default AdminAppointmnetDetailBoxAvatar
